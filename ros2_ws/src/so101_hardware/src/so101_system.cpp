// ros2_control system for the real SO-101 follower: STS3215 ticks <-> sim joint radians.
//
// read_only (default true): never writes to the servos, so the arm stays limp and commands are ignored.
// read_only false: on activate, goal = present position, torque limit set, torque on (the arm locks where
// it is). Each cycle the command is clamped to the joint limits and rate-limited before it is sent.
// If an arm joint falls far behind its target (blocked, overloaded) the driver holds the measured pose
// and ignores commands until they match that pose again (a new plan from the current state).
// Torque goes off on deactivate/exit: the arm goes limp and drops.
//
// Conversion follows LeRobot (lerobot/motors/motors_bus.py, use_degrees=True) with its calibration file:
//   arm:     deg = (raw - (min + max) / 2) * 360 / 4095;  rad = sign * deg2rad(deg - offset_deg)
//   gripper: g = clamp(raw, min, max) mapped to 0..100;    rad = closed_rad + g / 100 * (open_rad - closed_rad)
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "so101_hardware/sts_bus.hpp"

namespace so101_hardware
{

using CallbackReturn = hardware_interface::CallbackReturn;

class SO101System : public hardware_interface::SystemInterface
{
public:
  ~SO101System() override {torque_off();}

  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override
  {
    if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }
    const auto & hw = info.hardware_parameters;
    port_ = hw.at("port");
    read_only_ = hw.at("read_only") != "false";
    torque_limit_ = std::stoi(hw.at("torque_limit"));
    p_gain_ = std::stoi(hw.at("p_gain"));
    max_velocity_ = std::stod(hw.at("max_velocity"));
    max_tracking_error_ = std::stod(hw.at("max_tracking_error"));
    const std::string calib_file = hw.at("calibration_file");
    YAML::Node calib;  // LeRobot's JSON is valid YAML
    try {
      calib = YAML::LoadFile(calib_file);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(logger(), "cannot read calibration %s: %s", calib_file.c_str(), e.what());
      return CallbackReturn::ERROR;
    }

    for (const auto & j : info.joints) {
      Joint jt;
      const auto & p = j.parameters;
      jt.gripper = p.count("closed_rad") > 0;
      if (jt.gripper) {
        jt.closed_rad = std::stod(p.at("closed_rad"));
        jt.open_rad = std::stod(p.at("open_rad"));
      } else {
        jt.sign = std::stod(p.at("sign"));
        jt.offset_deg = std::stod(p.at("offset_deg"));
      }
      const auto & ci = j.command_interfaces.at(0);
      if (ci.min.empty() || ci.max.empty()) {
        RCLCPP_ERROR(logger(), "%s: position command interface needs min and max", j.name.c_str());
        return CallbackReturn::ERROR;
      }
      jt.lo = std::stod(ci.min);
      jt.hi = std::stod(ci.max);
      const auto c = calib[j.name];
      if (!c) {
        RCLCPP_ERROR(logger(), "joint %s missing from %s", j.name.c_str(), calib_file.c_str());
        return CallbackReturn::ERROR;
      }
      jt.id = static_cast<uint8_t>(c["id"].as<int>());
      jt.min = c["range_min"].as<int>();
      jt.max = c["range_max"].as<int>();
      if (c["drive_mode"].as<int>() != 0) {
        RCLCPP_ERROR(logger(), "%s: drive_mode != 0 is not handled", j.name.c_str());
        return CallbackReturn::ERROR;
      }
      joints_.push_back(jt);
      ids_.push_back(jt.id);
    }
    pos_.assign(joints_.size(), std::nan(""));
    vel_.assign(joints_.size(), 0.0);
    eff_.assign(joints_.size(), 0.0);
    cmd_.assign(joints_.size(), std::nan(""));
    target_.assign(joints_.size(), 0.0);
    return CallbackReturn::SUCCESS;
  }

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override
  {
    std::vector<hardware_interface::StateInterface> s;
    for (size_t i = 0; i < joints_.size(); ++i) {
      s.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &pos_[i]);
      s.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &vel_[i]);
      s.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &eff_[i]);
    }
    return s;
  }

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override
  {
    std::vector<hardware_interface::CommandInterface> c;
    for (size_t i = 0; i < joints_.size(); ++i) {
      c.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &cmd_[i]);
    }
    return c;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    if (!bus_.open(port_)) {
      RCLCPP_ERROR(logger(), "cannot open %s (in use by LeRobot or another launch? unplugged?)", port_.c_str());
      return CallbackReturn::ERROR;
    }
    // Check every servo answers before reporting anything.
    for (size_t i = 0; i < joints_.size(); ++i) {
      int raw = 0;
      if (!bus_.read_u16(joints_[i].id, StsBus::PRESENT_POSITION, raw)) {
        RCLCPP_ERROR(logger(), "servo %d (%s) does not answer: is the 12 V supply on?",
          joints_[i].id, info_.joints[i].name.c_str());
        bus_.close();
        return CallbackReturn::ERROR;
      }
      pos_[i] = to_rad(joints_[i], decode_position(raw));
      cmd_[i] = target_[i] = pos_[i];
      if (!read_only_) {
        // Goal = where the servo already is, so switching torque on does not move it.
        // Arm P gain: LeRobot stores 16 (sags ~0.3 rad per unit load); set after LOCK so it lasts only
        // until power-off and LeRobot's stored value is untouched. The gripper keeps LeRobot's settings.
        const uint8_t id = joints_[i].id;
        if (!bus_.write_u16(id, StsBus::GOAL_POSITION, raw) ||
          !bus_.write_u16(id, StsBus::TORQUE_LIMIT, torque_limit_) || !bus_.write_u8(id, StsBus::LOCK, 1) ||
          (!joints_[i].gripper && !bus_.write_u8(id, StsBus::P_COEFFICIENT, static_cast<uint8_t>(p_gain_))) ||
          !bus_.write_u8(id, StsBus::TORQUE_ENABLE, 1))
        {
          RCLCPP_ERROR(logger(), "servo %d (%s): enabling torque failed", id, info_.joints[i].name.c_str());
          torque_off();
          bus_.close();
          return CallbackReturn::ERROR;
        }
        torque_on_ = true;
      }
    }
    faulted_ = false;
    if (read_only_) {
      RCLCPP_INFO(logger(), "connected to %zu servos on %s (read-only: servos stay limp)",
        joints_.size(), port_.c_str());
    } else {
      RCLCPP_WARN(logger(), "connected to %zu servos on %s: TORQUE ON (limit %.0f%%, %.1f rad/s, arm P %d)",
        joints_.size(), port_.c_str(), torque_limit_ / 10.0, max_velocity_, p_gain_);
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    torque_off();
    bus_.close();
    return CallbackReturn::SUCCESS;
  }

  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override
  {
    std::vector<std::vector<uint8_t>> raw;  // position, speed, load: 2 bytes each
    if (!bus_.sync_read(StsBus::PRESENT_POSITION, 6, ids_, raw)) {
      // One bad packet now and then is normal on a serial bus; keep the last values.
      if (++failed_reads_ >= MAX_FAILED_READS) {
        RCLCPP_ERROR(logger(), "%d servo reads failed in a row", failed_reads_);
        return hardware_interface::return_type::ERROR;
      }
      return hardware_interface::return_type::OK;
    }
    failed_reads_ = 0;
    for (size_t i = 0; i < joints_.size(); ++i) {
      const auto & b = raw[i];
      const Joint & j = joints_[i];
      pos_[i] = to_rad(j, decode_position(b[0] | (b[1] << 8)));
      vel_[i] = to_rad_rate(j, decode_sign(b[2] | (b[3] << 8), 15));  // servo speed, ticks/s
      // Load: per mille of max torque, signed; reported as a fraction, in the joint's sim direction.
      eff_[i] = decode_sign(b[4] | (b[5] << 8), 10) / 1000.0 * (j.gripper ? 1.0 : j.sign);
    }
    return hardware_interface::return_type::OK;
  }

  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration & period) override
  {
    if (read_only_ || !torque_on_) {
      return hardware_interface::return_type::OK;
    }
    if (faulted_) {
      for (size_t i = 0; i < joints_.size(); ++i) {
        if (!joints_[i].gripper && !(std::abs(cmd_[i] - pos_[i]) < RESUME_TOLERANCE)) {
          return hardware_interface::return_type::OK;  // still holding (NaN commands hold too)
        }
      }
      RCLCPP_INFO(logger(), "commands match the measured pose again: resuming");
      faulted_ = false;
      target_ = pos_;
    }
    const double dt = std::clamp(period.seconds(), 0.001, 0.05);
    for (size_t i = 0; i < joints_.size(); ++i) {
      const Joint & j = joints_[i];
      const double goal = std::isnan(cmd_[i]) ? target_[i] : std::clamp(cmd_[i], j.lo, j.hi);
      // Gripper: no rate limit (weak: LeRobot set it to 50 % max torque, and the policy needs it fast)
      // and no tracking check (the cube blocks it on every grasp).
      if (j.gripper) {
        target_[i] = goal;
        continue;
      }
      const double step = max_velocity_ * dt;
      target_[i] += std::clamp(goal - target_[i], -step, step);
      if (std::abs(target_[i] - pos_[i]) > max_tracking_error_) {
        RCLCPP_ERROR(logger(), "%s at %.2f rad, target %.2f (blocked or overloaded?): holding the measured "
          "pose until a new plan starts from it", info_.joints[i].name.c_str(), pos_[i], target_[i]);
        faulted_ = true;
        target_ = pos_;
        break;
      }
    }
    std::vector<int> ticks(joints_.size());
    for (size_t i = 0; i < joints_.size(); ++i) {
      ticks[i] = encode_position(to_ticks(joints_[i], target_[i]));
    }
    bus_.sync_write_u16(StsBus::GOAL_POSITION, ids_, ticks);
    return hardware_interface::return_type::OK;
  }

private:
  struct Joint
  {
    uint8_t id = 0;
    int min = 0, max = 4095;
    bool gripper = false;
    double sign = 1.0, offset_deg = 0.0;
    double closed_rad = 0.0, open_rad = 0.0;
    double lo = 0.0, hi = 0.0;  // joint limits (rad) for commands
  };

  static double to_rad(const Joint & j, int raw)
  {
    if (j.gripper) {
      const double g = (std::clamp(raw, j.min, j.max) - j.min) / static_cast<double>(j.max - j.min);
      return j.closed_rad + g * (j.open_rad - j.closed_rad);
    }
    const double deg = (raw - (j.min + j.max) / 2.0) * 360.0 / 4095.0;
    return j.sign * (deg - j.offset_deg) * M_PI / 180.0;
  }

  static double to_rad_rate(const Joint & j, int ticks_per_s)
  {
    if (j.gripper) {
      return ticks_per_s * (j.open_rad - j.closed_rad) / (j.max - j.min);
    }
    return j.sign * ticks_per_s * 2.0 * M_PI / 4095.0;
  }

  static int to_ticks(const Joint & j, double rad)
  {
    if (j.gripper) {
      const double g = (rad - j.closed_rad) / (j.open_rad - j.closed_rad);
      return static_cast<int>(std::lround(j.min + g * (j.max - j.min)));
    }
    const double deg = j.sign * rad * 180.0 / M_PI + j.offset_deg;
    return static_cast<int>(std::lround(deg * 4095.0 / 360.0 + (j.min + j.max) / 2.0));
  }

  void torque_off()
  {
    if (!torque_on_) {
      return;
    }
    for (size_t i = 0; i < joints_.size(); ++i) {
      if (!bus_.write_u8(joints_[i].id, StsBus::TORQUE_ENABLE, 0) ||
        !bus_.write_u8(joints_[i].id, StsBus::LOCK, 0))
      {
        RCLCPP_ERROR(logger(), "servo %d: torque off failed: switch off the 12 V supply", joints_[i].id);
      }
    }
    torque_on_ = false;
    RCLCPP_WARN(logger(), "torque off: the arm is limp");
  }

  static rclcpp::Logger logger() {return rclcpp::get_logger("SO101System");}

  static constexpr int MAX_FAILED_READS = 10;
  static constexpr double RESUME_TOLERANCE = 0.05;  // rad
  std::string port_;
  bool read_only_ = true, torque_on_ = false, faulted_ = false;
  int torque_limit_ = 500, p_gain_ = 32;
  double max_velocity_ = 2.5, max_tracking_error_ = 0.5;
  StsBus bus_;
  std::vector<Joint> joints_;
  std::vector<uint8_t> ids_;
  std::vector<double> pos_, vel_, eff_, cmd_, target_;
  int failed_reads_ = 0;
};

}  // namespace so101_hardware

PLUGINLIB_EXPORT_CLASS(so101_hardware::SO101System, hardware_interface::SystemInterface)

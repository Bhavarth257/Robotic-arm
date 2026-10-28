// ros2_control system for the real SO-101 follower: STS3215 ticks -> sim joint radians.
// READ-ONLY for now: it never writes to the servos (no torque, no goal positions), so the arm stays limp
// and commands from controllers are ignored. Writing comes with the first powered test (M8a).
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
  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override
  {
    if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }
    port_ = info.hardware_parameters.at("port");
    const std::string calib_file = info.hardware_parameters.at("calibration_file");
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
    cmd_.assign(joints_.size(), std::nan(""));
    return CallbackReturn::SUCCESS;
  }

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override
  {
    std::vector<hardware_interface::StateInterface> s;
    for (size_t i = 0; i < joints_.size(); ++i) {
      s.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &pos_[i]);
      s.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &vel_[i]);
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
      RCLCPP_ERROR(logger(), "cannot open %s (in use by LeRobot? unplugged?)", port_.c_str());
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
      cmd_[i] = pos_[i];
    }
    RCLCPP_INFO(logger(), "connected to %zu servos on %s (read-only: servos stay limp)",
      joints_.size(), port_.c_str());
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    bus_.close();
    return CallbackReturn::SUCCESS;
  }

  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration & period) override
  {
    std::vector<int> raw;
    if (!bus_.sync_read_u16(StsBus::PRESENT_POSITION, ids_, raw)) {
      // One bad packet now and then is normal on a serial bus; keep the last values.
      if (++failed_reads_ >= MAX_FAILED_READS) {
        RCLCPP_ERROR(logger(), "%d servo reads failed in a row", failed_reads_);
        return hardware_interface::return_type::ERROR;
      }
      return hardware_interface::return_type::OK;
    }
    failed_reads_ = 0;
    const double dt = period.seconds();
    for (size_t i = 0; i < joints_.size(); ++i) {
      const double q = to_rad(joints_[i], decode_position(raw[i]));
      vel_[i] = dt > 0 ? (q - pos_[i]) / dt : 0.0;
      pos_[i] = q;
    }
    return hardware_interface::return_type::OK;
  }

  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override
  {
    return hardware_interface::return_type::OK;  // read-only
  }

private:
  struct Joint
  {
    uint8_t id = 0;
    int min = 0, max = 4095;
    bool gripper = false;
    double sign = 1.0, offset_deg = 0.0;
    double closed_rad = 0.0, open_rad = 0.0;
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

  static rclcpp::Logger logger() {return rclcpp::get_logger("SO101System");}

  static constexpr int MAX_FAILED_READS = 10;
  std::string port_;
  StsBus bus_;
  std::vector<Joint> joints_;
  std::vector<uint8_t> ids_;
  std::vector<double> pos_, vel_, cmd_;
  int failed_reads_ = 0;
};

}  // namespace so101_hardware

PLUGINLIB_EXPORT_CLASS(so101_hardware::SO101System, hardware_interface::SystemInterface)

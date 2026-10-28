#include "so101_hardware/sts_bus.hpp"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>

namespace so101_hardware
{
namespace
{
constexpr uint8_t BROADCAST = 0xFE;
constexpr uint8_t INST_READ = 0x02;
constexpr uint8_t INST_SYNC_READ = 0x82;

speed_t to_speed(int baud)
{
  switch (baud) {
    case 1000000: return B1000000;
    case 500000: return B500000;
    case 115200: return B115200;
    default: return B0;
  }
}
}  // namespace

StsBus::~StsBus() {close();}

bool StsBus::open(const std::string & port, int baud)
{
  close();
  const speed_t speed = to_speed(baud);
  if (speed == B0) {
    return false;
  }
  fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    return false;
  }
  termios tio{};
  cfmakeraw(&tio);
  tio.c_cflag |= CLOCAL | CREAD;
  cfsetispeed(&tio, speed);
  cfsetospeed(&tio, speed);
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    close();
    return false;
  }
  tcflush(fd_, TCIOFLUSH);
  return true;
}

void StsBus::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

// Packet: FF FF id len instr params... checksum, len = params + 2, checksum = ~(id + len + instr + params).
bool StsBus::send(uint8_t id, uint8_t instr, const std::vector<uint8_t> & params)
{
  std::vector<uint8_t> pkt{0xFF, 0xFF, id, static_cast<uint8_t>(params.size() + 2), instr};
  pkt.insert(pkt.end(), params.begin(), params.end());
  uint8_t sum = 0;
  for (size_t i = 2; i < pkt.size(); ++i) {
    sum += pkt[i];
  }
  pkt.push_back(static_cast<uint8_t>(~sum));
  tcflush(fd_, TCIFLUSH);  // drop stale bytes from an earlier failed read
  return ::write(fd_, pkt.data(), pkt.size()) == static_cast<ssize_t>(pkt.size());
}

bool StsBus::read_bytes(uint8_t * buf, size_t n)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
  size_t got = 0;
  while (got < n) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now()).count();
    if (left <= 0) {
      return false;
    }
    pollfd p{fd_, POLLIN, 0};
    if (poll(&p, 1, static_cast<int>(left)) <= 0) {
      return false;
    }
    const ssize_t r = ::read(fd_, buf + got, n - got);
    if (r <= 0) {
      return false;
    }
    got += static_cast<size_t>(r);
  }
  return true;
}

// Status packet: FF FF id len error data... checksum.
bool StsBus::receive(uint8_t id, size_t n_data, std::vector<uint8_t> & data)
{
  uint8_t b = 0;
  int ff = 0;
  while (ff < 2) {  // resync on the header
    if (!read_bytes(&b, 1)) {
      return false;
    }
    ff = (b == 0xFF) ? ff + 1 : 0;
  }
  uint8_t head[3];  // id, len, error
  if (!read_bytes(head, 3) || head[0] != id || head[1] != n_data + 2) {
    return false;
  }
  data.resize(n_data + 1);  // + checksum
  if (!read_bytes(data.data(), data.size())) {
    return false;
  }
  uint8_t sum = head[0] + head[1] + head[2];
  for (size_t i = 0; i < n_data; ++i) {
    sum += data[i];
  }
  if (static_cast<uint8_t>(~sum) != data[n_data]) {
    return false;
  }
  data.resize(n_data);
  return true;
}

bool StsBus::sync_read_u16(uint8_t addr, const std::vector<uint8_t> & ids, std::vector<int> & out)
{
  std::vector<uint8_t> params{addr, 2};
  params.insert(params.end(), ids.begin(), ids.end());
  if (!send(BROADCAST, INST_SYNC_READ, params)) {
    return false;
  }
  out.assign(ids.size(), 0);
  std::vector<uint8_t> data;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!receive(ids[i], 2, data)) {
      return false;
    }
    out[i] = data[0] | (data[1] << 8);
  }
  return true;
}

bool StsBus::read_u16(uint8_t id, uint8_t addr, int & out)
{
  std::vector<uint8_t> data;
  if (!send(id, INST_READ, {addr, 2}) || !receive(id, 2, data)) {
    return false;
  }
  out = data[0] | (data[1] << 8);
  return true;
}

}  // namespace so101_hardware

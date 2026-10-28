// Minimal Feetech STS3215 serial bus (protocol 0, little-endian registers, 1 Mbaud).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace so101_hardware
{

class StsBus
{
public:
  ~StsBus();
  bool open(const std::string & port, int baud = 1000000);
  void close();
  bool is_open() const {return fd_ >= 0;}

  // One read of a 2-byte register from every id (SYNC READ). False on timeout or a bad packet.
  bool sync_read_u16(uint8_t addr, const std::vector<uint8_t> & ids, std::vector<int> & out);
  bool read_u16(uint8_t id, uint8_t addr, int & out);

  static constexpr uint8_t PRESENT_POSITION = 56;

private:
  bool send(uint8_t id, uint8_t instr, const std::vector<uint8_t> & params);
  bool receive(uint8_t id, size_t n_data, std::vector<uint8_t> & data);
  bool read_bytes(uint8_t * buf, size_t n);

  int fd_ = -1;
  int timeout_ms_ = 10;
};

// Present_Position is sign-magnitude with the sign in bit 15.
inline int decode_position(int raw)
{
  return (raw & 0x8000) ? -(raw & 0x7fff) : raw;
}

}  // namespace so101_hardware

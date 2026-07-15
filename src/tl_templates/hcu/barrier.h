#pragma once

// Hygon gfx946 ABarrier / EBarrier wrappers (hardware slot id in [0, 15]).

namespace tl {

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx946__)

TL_DEVICE void abarrier_init(int abar_id, int wave_cnt) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_abarrier_init(abar_id, wave_cnt);
#endif
}

TL_DEVICE void abarrier_inv(int abar_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_abarrier_inv(abar_id);
#endif
}

TL_DEVICE int abarrier_arrive(int abar_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_hcu_s_abarrier_arrive(abar_id);
#else
  return 0;
#endif
}

TL_DEVICE int abarrier_arrive_cnt(int abar_id, int wave_cnt) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_hcu_s_abarrier_arrive_cnt(abar_id, wave_cnt);
#else
  return 0;
#endif
}

// Blocking wait: incomplete phases block up to the hardware default suspend
// timeout (2^16 * 4 cycles). Returns non-zero when the phase is complete.
TL_DEVICE int abarrier_try_wait(int abar_id, int phase) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_hcu_s_abarrier_try_wait(abar_id, phase);
#else
  return 0;
#endif
}

// Non-blocking poll: returns immediately without blocking the wave.
TL_DEVICE int abarrier_test_wait(int abar_id, int phase) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_hcu_s_abarrier_test_wait(abar_id, phase);
#else
  return 0;
#endif
}

// Wait-until-complete helper (mainly for debug/maint). Loops on try_wait until
// the phase completes.
TL_DEVICE void abarrier_wait(int abar_id, int phase) {
#if defined(__HIP_DEVICE_COMPILE__)
  while (__builtin_hcu_s_abarrier_try_wait(abar_id, phase) == 0) {
  }
#endif
}

TL_DEVICE void abarrier_seq(int abar_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_abarrier_seq(abar_id);
#endif
}

TL_DEVICE void abarrier_expect_tx(int abar_id, int tx_cnt) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_abarrier_expect_tx(abar_id, tx_cnt);
#endif
}

TL_DEVICE void abarrier_complete_tx(int abar_id, int tx_cnt) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_abarrier_complete_tx(abar_id, tx_cnt);
#endif
}

TL_DEVICE void ebarrier_sync(int ebar_id) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_ebarrier_sync(ebar_id);
#endif
}

TL_DEVICE void ebarrier_sync_cnt(int ebar_id, int wave_cnt) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_ebarrier_sync_cnt(ebar_id, wave_cnt);
#endif
}

TL_DEVICE void ebarrier_arrive(int ebar_id, int wave_cnt) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_hcu_s_ebarrier_arrive_cnt(ebar_id, wave_cnt);
#endif
}

#else

// Non-gfx946 device: no definitions. Include-only is fine; emitted
// tl::abarrier_* / tl::ebarrier_* call sites fail at compile time (undefined in
// namespace tl).

#endif // !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx946__)

} // namespace tl

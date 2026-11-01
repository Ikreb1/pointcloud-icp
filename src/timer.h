#pragma once

#include <chrono>

// Milliseconds elapsed since `start`. steady_clock never jumps (unaffected by
// system clock changes), so it is the right clock for measuring durations.
inline double msSince(std::chrono::steady_clock::time_point start)
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

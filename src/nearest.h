#pragma once

#include <limits>

#include "cloud.h"

struct NearestResult
{
	int index = -1;
	float distanceSquared = std::numeric_limits<float>::max();
};

// Brute force: checks every point in `target`. O(N) per query. The baseline
// every other search is checked and benchmarked against.
NearestResult nearestNeighbour(const Cloud& target, const Point& query);

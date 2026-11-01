#include "nearest.h"

// Compares squared distances, since sqrt preserves order and is expensive.
// noinline: keeps brute force as its own entry in perf report instead of being
// folded into its caller. One call per 10k distance checks, so the cost is negligible.
[[gnu::noinline]] NearestResult nearestNeighbour(const Cloud& target, const Point& query)
{
	NearestResult best;
	for (int i = 0; i < static_cast<int>(target.points.size()); ++i)
	{
		const Point& p = target.points[i];
		const float dx = p.x - query.x;
		const float dy = p.y - query.y;
		const float dz = p.z - query.z;
		const float d2 = dx * dx + dy * dy + dz * dz;
		if (d2 < best.distanceSquared)
		{
			best.distanceSquared = d2;
			best.index = i;
		}
	}
	return best;
}

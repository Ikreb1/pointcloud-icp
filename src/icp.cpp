#include "icp.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <optional>

#include "kdtree.h"
#include "nearest.h"
#include "timer.h"

IcpResult runIcp(const Cloud& source, const Cloud& target, Search search, int maxIterations, float tolerance, bool verbose)
{
	IcpResult result;

	// The target never moves, so the tree is built once and reused every iteration.
	const auto buildStart = std::chrono::steady_clock::now();
	std::optional<KdTree> tree;
	if (search != Search::BruteForce)
		tree.emplace(target);
	result.buildMs = msSince(buildStart);

	double totalMs = 0, matchMs = 0, alignMs = 0, transformMs = 0;
	Cloud current = source;
	float previousRmse = std::numeric_limits<float>::max();

	for (int iter = 0; iter < maxIterations; ++iter)
	{
		const auto start = std::chrono::steady_clock::now();
		auto phase = start;

		// Match: nearest target point for every source point, and add each pair
		// straight into the alignment sums (no separate pass over the pairs).
		// Brute force is O(N*M), the tree roughly O(N log M).
		// Threads only read the tree/target. The two totals are `reduction`s: each
		// thread fills its own private copy, OpenMP adds them up at the end.
		double sumSquared = 0;
		PairSums sums;
#pragma omp parallel for reduction(+ : sumSquared, sums) schedule(static)
		for (int i = 0; i < static_cast<int>(current.points.size()); ++i)
		{
			const Point& query = current.points[i];
			const NearestResult nearest = search == Search::KdTreeSimd ? tree->nearest(query)
			                            : search == Search::KdTreeScalar ? tree->nearestScalar(query)
			                            : nearestNeighbour(target, query);
			sumSquared += nearest.distanceSquared;
			sums.add(query, target.points[nearest.index]);
		}
		const float rmse = static_cast<float>(std::sqrt(sumSquared / current.points.size()));
		matchMs += msSince(phase);

		// Align.
		phase = std::chrono::steady_clock::now();
		const RigidTransform step = rigidTransformFromSums(sums);
		alignMs += msSince(phase);

		// Apply.
		phase = std::chrono::steady_clock::now();
		transformInPlace(current, step.R, step.t);
		transformMs += msSince(phase);

		// Compose: total = step after total, i.e. p -> R_s (R p + t) + t_s.
		result.transform.R = step.R * result.transform.R;
		result.transform.t = step.R * result.transform.t + step.t;

		const double ms = msSince(start);
		totalMs += ms;
		if (verbose)
			std::cout << "iter " << iter << "  rmse " << rmse << "  " << ms << " ms\n";

		result.iterations = iter + 1;
		result.rmse = rmse;
		if (previousRmse - rmse < tolerance)
			break;
		previousRmse = rmse;
	}
	result.msPerIteration = totalMs / result.iterations;
	result.matchMs = matchMs / result.iterations;
	result.alignMs = alignMs / result.iterations;
	result.transformMs = transformMs / result.iterations;
	return result;
}

#pragma once

#include "alignment.h"
#include "cloud.h"

enum class Search
{
	BruteForce,
	KdTreeScalar,
	KdTreeSimd,
};

struct IcpResult
{
	RigidTransform transform; // accumulated over all iterations: source -> target
	int iterations = 0;
	float rmse = 0;          // root mean square distance between matched pairs
	double msPerIteration = 0;
	double buildMs = 0;      // k-d tree build, once per run

	// Per-iteration averages of each phase, in ms.
	double matchMs = 0;      // nearest neighbour + pair sums for every point (parallel)
	double alignMs = 0;      // centroids, H and SVD from the pair sums (serial, tiny)
	double transformMs = 0;  // apply R, t to the moving cloud (parallel)
};

// Point-to-point ICP: repeatedly match each source point to its nearest target
// point, solve for the best rigid transform, and apply it. Stops when the error
// improves by less than `tolerance` or after `maxIterations`.
IcpResult runIcp(const Cloud& source, const Cloud& target, Search search, int maxIterations, float tolerance, bool verbose);

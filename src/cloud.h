#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>

struct Point
{
	float x = 0, y = 0, z = 0;

	void randomize(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		x = dist(rng);
		y = dist(rng);
		z = dist(rng);
	}
};

struct Cloud
{
	std::vector<Point> points;
};

// `count` points uniformly inside the cube [-1, 1]^3.
Cloud makeRandomCloud(int count, std::mt19937& rng);

// Returns a copy of `in` with Gaussian noise (standard deviation `sigma`) added to every coordinate.
Cloud addNoise(const Cloud& in, float sigma, std::mt19937& rng);

// Rotation of `radians` around the z axis.
Eigen::Matrix3f rotationZ(float radians);

// Returns a copy of `in` with p' = R * p + t applied to every point.
Cloud transformCloud(const Cloud& in, const Eigen::Matrix3f& R, const Eigen::Vector3f& t);

// Applies p' = R * p + t to every point of `cloud`, in place and in parallel.
// Used inside ICP: no new allocation each iteration, and each point is independent.
void transformInPlace(Cloud& cloud, const Eigen::Matrix3f& R, const Eigen::Vector3f& t);

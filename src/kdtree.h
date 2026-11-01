#pragma once

#include <vector>

#include "cloud.h"
#include "nearest.h"

// Max points per k-d tree leaf. Tweak and benchmark: smaller = deeper tree and
// more node visits, larger = longer brute-force scans in each leaf.
constexpr int KD_LEAF_SIZE = 32;
static_assert(KD_LEAF_SIZE % 8 == 0, "leaves must be whole 8-wide SIMD registers");

// k-d tree for nearest-neighbour queries. Built once per target cloud (the
// target never moves during ICP), then queried for every source point.
//
// Layout:
// - Nodes live in one flat vector and refer to children by index.
// - Points are stored as Structure of Arrays (separate x, y, z arrays),
//   reordered so each leaf's points are contiguous. One AVX2 load then
//   fills a register with 8 x values of the same leaf.
// - Every leaf starts at a multiple of 8 and holds a multiple of 8 points
//   (except the very last leaf), so leaves are whole SIMD registers. The
//   arrays are padded to a multiple of 8 with points at infinity, which are
//   never the nearest.
class KdTree
{
public:
	explicit KdTree(const Cloud& cloud, int leafSize = KD_LEAF_SIZE);

	// Leaf scans with AVX2: 8 points per instruction.
	NearestResult nearest(const Point& query) const;

	// Same tree and layout, leaf scans one point at a time. For benchmarking.
	NearestResult nearestScalar(const Point& query) const;

	static constexpr int SIMD_WIDTH = 8; // floats per 256-bit AVX2 register

private:
	struct Node
	{
		int axis;    // 0/1/2 = split axis, -1 = leaf
		float split; // interior: points with coord <= split go left, >= split go right
		int a, b;    // interior: left/right child node. leaf: [a, b) range in the point arrays
	};

	int build(const Cloud& cloud, std::vector<int>& order, int begin, int end);
	void scanLeafScalar(const Node& node, const Point& query, NearestResult& best) const;
	void scanLeafSimd(const Node& node, const Point& query, NearestResult& best) const;

	template <bool UseSimd>
	void search(const Point& query, NearestResult& best) const;

	int m_leafSize;
	std::vector<Node> m_nodes;
	std::vector<float> m_x, m_y, m_z;
	std::vector<int> m_originalIndex;
};

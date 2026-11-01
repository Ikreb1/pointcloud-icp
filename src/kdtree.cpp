#include "kdtree.h"

#include <algorithm>
#include <bit>
#include <limits>

#include <immintrin.h>

namespace
{

float coord(const Point& p, int axis)
{
	return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

#ifdef __AVX2__
// Smallest of the 8 floats in a 256-bit register ("horizontal min").
inline float horizontalMin(__m256 v)
{
	__m128 m = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)); // 8 -> 4
	m = _mm_min_ps(m, _mm_movehl_ps(m, m));                                          // 4 -> 2
	m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));                                      // 2 -> 1
	return _mm_cvtss_f32(m);
}
#endif

} // namespace

KdTree::KdTree(const Cloud& cloud, int leafSize)
	: m_leafSize(leafSize)
{
	const int n = static_cast<int>(cloud.points.size());
	std::vector<int> order(n);
	for (int i = 0; i < n; ++i)
		order[i] = i;

	m_nodes.reserve(2 * n / leafSize + 1);
	build(cloud, order, 0, n);

	const int padded = (n + SIMD_WIDTH - 1) / SIMD_WIDTH * SIMD_WIDTH;
	const float inf = std::numeric_limits<float>::infinity();
	m_x.assign(padded, inf);
	m_y.assign(padded, inf);
	m_z.assign(padded, inf);
	m_originalIndex.assign(padded, -1);
	for (int i = 0; i < n; ++i)
	{
		const Point& p = cloud.points[order[i]];
		m_x[i] = p.x;
		m_y[i] = p.y;
		m_z[i] = p.z;
		m_originalIndex[i] = order[i];
	}
}

NearestResult KdTree::nearest(const Point& query) const
{
	NearestResult best;
	search<true>(query, best);
	return best;
}

NearestResult KdTree::nearestScalar(const Point& query) const
{
	NearestResult best;
	search<false>(query, best);
	return best;
}

// Builds the subtree for order[begin, end) and returns its node index.
int KdTree::build(const Cloud& cloud, std::vector<int>& order, int begin, int end)
{
	const int nodeIndex = static_cast<int>(m_nodes.size());
	m_nodes.push_back({});

	if (end - begin <= m_leafSize)
	{
		m_nodes[nodeIndex] = {-1, 0.0f, begin, end};
		return nodeIndex;
	}

	// Split on the axis where this subset is widest.
	Eigen::Vector3f lo = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
	Eigen::Vector3f hi = -lo;
	for (int i = begin; i < end; ++i)
	{
		const Point& p = cloud.points[order[i]];
		const Eigen::Vector3f v(p.x, p.y, p.z);
		lo = lo.cwiseMin(v);
		hi = hi.cwiseMax(v);
	}
	int axis;
	(hi - lo).maxCoeff(&axis);

	// Split near the median, rounded up to a multiple of 8 points. `begin`
	// starts at 0 and every left half is a multiple of 8, so every node starts
	// on a multiple of 8: leaves are whole SIMD registers.
	const int half = (end - begin) / 2;
	const int mid = begin + (half + SIMD_WIDTH - 1) / SIMD_WIDTH * SIMD_WIDTH;

	// nth_element puts the element that belongs at `mid` there, smaller-or-equal
	// before it, larger-or-equal after it. O(n), no full sort needed.
	std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
		[&](int l, int r) { return coord(cloud.points[l], axis) < coord(cloud.points[r], axis); });
	const float split = coord(cloud.points[order[mid]], axis);

	// Children are built after this node was pushed, so m_nodes may have
	// reallocated: write through the index, never hold a reference across build().
	const int left = build(cloud, order, begin, mid);
	const int right = build(cloud, order, mid, end);
	m_nodes[nodeIndex] = {axis, split, left, right};
	return nodeIndex;
}

void KdTree::scanLeafScalar(const Node& node, const Point& query, NearestResult& best) const
{
	for (int i = node.a; i < node.b; ++i)
	{
		const float dx = m_x[i] - query.x;
		const float dy = m_y[i] - query.y;
		const float dz = m_z[i] - query.z;
		const float d2 = dx * dx + dy * dy + dz * dz;
		if (d2 < best.distanceSquared)
		{
			best.distanceSquared = d2;
			best.index = m_originalIndex[i];
		}
	}
}

void KdTree::scanLeafSimd(const Node& node, const Point& query, NearestResult& best) const
{
#ifdef __AVX2__
	// Broadcast: the query's x copied into all 8 lanes, same for y and z.
	const __m256 qx = _mm256_set1_ps(query.x);
	const __m256 qy = _mm256_set1_ps(query.y);
	const __m256 qz = _mm256_set1_ps(query.z);

	// Blocks of 8 points. May read past node.b into the next leaf or the
	// infinity padding; both are harmless (real points are just extra
	// candidates, padding is never closer).
	for (int i = node.a; i < node.b; i += SIMD_WIDTH)
	{
		const __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(&m_x[i]), qx);
		const __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(&m_y[i]), qy);
		const __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(&m_z[i]), qz);
		__m256 d2 = _mm256_mul_ps(dx, dx);
		d2 = _mm256_fmadd_ps(dy, dy, d2); // d2 += dy * dy, in one instruction
		d2 = _mm256_fmadd_ps(dz, dz, d2);

		// Usually no point in the block beats the best so far: one compare
		// and move on. Only when one does, find which lane it was.
		const float blockMin = horizontalMin(d2);
		if (blockMin < best.distanceSquared)
		{
			const __m256 isMin = _mm256_cmp_ps(d2, _mm256_set1_ps(blockMin), _CMP_EQ_OQ);
			const int lane = std::countr_zero(static_cast<unsigned>(_mm256_movemask_ps(isMin)));
			best.distanceSquared = blockMin;
			best.index = m_originalIndex[i + lane];
		}
	}
#else
	// No AVX2 (e.g. ARM): fall back to scalar until there is a NEON version.
	scanLeafScalar(node, query, best);
#endif
}

// Iterative search with an explicit stack instead of recursion: no function
// call per node visited. Walk down to the leaf the query is in, pushing each
// far side we pass along with its distance to the split plane. After a leaf,
// pop far sides until one could still hold something closer than `best`.
template <bool UseSimd>
void KdTree::search(const Point& query, NearestResult& best) const
{
	// The query as an array, so the split axis is an index instead of a
	// chain of branches (coord()).
	const float q[3] = {query.x, query.y, query.z};

	// Depth is about log2(points / leaf size): 64 covers far more points than fit in memory.
	constexpr int MAX_DEPTH = 64;
	int stackNode[MAX_DEPTH];
	float stackPlaneDistSq[MAX_DEPTH];
	int top = 0;

	int nodeIndex = 0;
	while (true)
	{
		// Down to a leaf, always taking the side the query is on.
		const Node* node = &m_nodes[nodeIndex];
		while (node->axis >= 0)
		{
			const float diff = q[node->axis] - node->split;
			const int nearChild = diff < 0 ? node->a : node->b;
			const int farChild = diff < 0 ? node->b : node->a;
			stackNode[top] = farChild;
			stackPlaneDistSq[top] = diff * diff;
			++top;
			node = &m_nodes[nearChild];
		}

		if constexpr (UseSimd)
			scanLeafSimd(*node, query, best);
		else
			scanLeafScalar(*node, query, best);

		// Backtrack: everything on a far side is at least planeDist away. `best`
		// has usually shrunk since it was pushed, so most far sides are skipped
		// here without ever being visited.
		do
		{
			if (top == 0)
				return;
			--top;
		} while (stackPlaneDistSq[top] >= best.distanceSquared);
		nodeIndex = stackNode[top];
	}
}

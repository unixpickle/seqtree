package seqtree

import (
	"math"
	"math/rand"
)

type Clusters struct {
	Deltas  [][]float32
	Centers [][]float32
}

// Evaluate finds the output delta corresponding to the
// closest center to v.
func (c *Clusters) Evaluate(v []float32) []float32 {
	idx, _ := c.Find(v)
	return c.Deltas[idx]
}

// Find finds the closest center index to v and also
// returns the squared distance.
func (c *Clusters) Find(v []float32) (int, float32) {
	minDist := float32(math.Inf(1))
	minIdx := 0
	for i, center := range c.Centers {
		dist := vectorNormSquared(vectorDifference(v, center))
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx, minDist
}

// KMeans builds clusters.
type KMeans struct {
	NumClusters   int
	MaxIterations int
}

// Cluster clusters the data points into centers.
func (k *KMeans) Cluster(data [][]float32) [][]float32 {
	result := k.initialize(data)
	// TODO: early stopping if converges.
	for i := 0; i < k.MaxIterations; i++ {
		result = k.iterate(data, result)
	}
	return result
}

func (k *KMeans) initialize(data [][]float32) [][]float32 {
	var result [][]float32

	// Random initial center.
	result = append(result, data[rand.Intn(len(data))])

	// Use k-means++ to sample remaining centers.
	for len(result) < k.NumClusters {
		var sqDists []float32
		totalDist := newKahanSum(1)
		for _, x := range data {
			_, dist := (&Clusters{Centers: result}).Find(x)
			sqDists = append(sqDists, dist)
			totalDist.Add([]float32{dist})
		}

		p := rand.Float32() * totalDist.Sum()[0]

		totalDist = newKahanSum(1)
		for i, dist := range sqDists {
			totalDist.Add([]float32{dist})
			if totalDist.Sum()[0] >= p {
				result = append(result, data[i])
				break
			}
		}
		if totalDist.Sum()[0] < p {
			// Rare edge case, likely due to rounding.
			result = append(result, data[len(data)-1])
		}
	}

	return result
}

func (k *KMeans) iterate(data, centers [][]float32) [][]float32 {
	dim := len(centers[0])

	counts := make([]int, len(centers))
	sums := make([]*kahanSum, len(centers))
	for i := range sums {
		sums[i] = newKahanSum(dim)
	}

	c := &Clusters{Centers: centers}
	for _, x := range data {
		i, _ := c.Find(x)
		sums[i].Add(x)
		counts[i] += 1
	}

	res := make([][]float32, len(sums))
	for i, s := range sums {
		res[i] = make([]float32, dim)
		for j, x := range s.Sum() {
			res[i][j] = x / float32(counts[i])
		}
	}

	return res
}

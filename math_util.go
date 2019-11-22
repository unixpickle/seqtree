package seqtree

import "math"

// minimizeUnary minimizes a function of one variable
// along an interval.
//
// This is only guaranteed to find a local minimum, not a
// global minimum.
func minimizeUnary(minX, maxX float32, iters int, f func(x float32) float32) float32 {
	// Golden section search:
	// https://en.wikipedia.org/wiki/Golden-section_search

	var midValue1, midValue2 *float32

	for i := 0; i < iters; i++ {
		mid1 := maxX - (maxX-minX)/math.Phi
		mid2 := minX + (maxX-minX)/math.Phi
		if midValue1 == nil {
			x := f(mid1)
			midValue1 = &x
		}
		if midValue2 == nil {
			x := f(mid2)
			midValue2 = &x
		}

		if *midValue2 < *midValue1 {
			minX = mid1
			midValue1 = midValue2
			midValue2 = nil
		} else {
			maxX = mid2
			midValue2 = midValue1
			midValue1 = nil
		}
	}

	return (minX + maxX) / 2
}

func vectorNormSquared(v []float32) float32 {
	var res float32
	for _, x := range v {
		res += x * x
	}
	return res
}

func gradientSum(ts []*TimestepSample, dim int) []float32 {
	if dim == 0 {
		dim = len(ts[0].Timestep().Gradient)
	}
	sum := newKahanSum(dim)
	for _, t := range ts {
		sum.Add(t.Timestep().Gradient)
	}
	return sum.Sum()
}

func gradientMean(ts []*TimestepSample) []float32 {
	sum := gradientSum(ts, 0)
	scale := 1 / float32(len(ts))
	for i := range sum {
		sum[i] *= scale
	}
	return sum
}

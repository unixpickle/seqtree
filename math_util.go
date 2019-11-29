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

func addDelta(v1, v2 []float32, scale float32) []float32 {
	res := make([]float32, len(v1))
	for i, x := range v1 {
		res[i] = x + v2[i]*scale
	}
	return res
}

func vectorDifference(v1, v2 []float32) []float32 {
	res := make([]float32, len(v1))
	for i, x := range v1 {
		res[i] = x - v2[i]
	}
	return res
}

func vectorDot(v1, v2 []float32) float32 {
	var res float32
	for i, x := range v1 {
		res += x * v2[i]
	}
	return res
}

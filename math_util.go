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

type SquareMatrix struct {
	Dim  int
	Data []float32
}

func (s *SquareMatrix) Inverse() *SquareMatrix {
	if s.Dim == 1 {
		return &SquareMatrix{
			Dim:  s.Dim,
			Data: []float32{1 / s.Data[0]},
		}
	} else if s.Dim == 2 {
		scale := 1 / (s.Data[0]*s.Data[3] - s.Data[1]*s.Data[2])
		return &SquareMatrix{
			Dim: s.Dim,
			Data: []float32{
				scale * s.Data[3],
				-scale * s.Data[1],
				-scale * s.Data[2],
				scale * s.Data[0],
			},
		}
	}
	panic("inverse not implemented for larger matrices")
}

func (s *SquareMatrix) VectorProduct(v []float32) []float32 {
	if len(v) != s.Dim {
		panic("dimension mismatch")
	}
	res := make([]float32, len(v))
	for row := 0; row < s.Dim; row++ {
		start := row * s.Dim
		for i, x := range v {
			res[row] += x * s.Data[start+i]
		}
	}
	return res
}

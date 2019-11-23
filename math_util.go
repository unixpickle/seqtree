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

type polynomial []float32

// newPolynomialLogSigmoid approximates the log of the
// sigmoid function as a high-order polynomial.
// The resulting polynomial is p(a) ~= log(sigmoid(x+a)).
// It is very accurate when abs(a) < 1.
func newPolynomialLogSigmoid(x float32) polynomial {
	// Create a short Taylor series.
	exp := float32(math.Exp(float64(x)))
	invExp := float32(math.Exp(float64(-x)))
	if math.IsInf(float64(exp), 1) {
		exp = float32(math.Exp(12))
	} else if math.IsInf(float64(invExp), 1) {
		invExp = float32(math.Exp(12))
	}

	logValue := x
	if x > -12 {
		logValue = float32(math.Log(float64(1 / (1 + invExp))))
	}

	exp2 := exp * exp
	exp3 := exp2 * exp
	exp4 := exp2 * exp2
	exp5 := exp3 * exp2
	exp6 := exp3 * exp3

	expP := exp + 1
	expP2 := expP * expP
	expP3 := expP2 * expP
	expP4 := expP2 * expP2
	expP5 := expP4 * expP
	expP6 := expP3 * expP3
	expP7 := expP4 * expP3
	expP8 := expP4 * expP4

	return polynomial{
		logValue,
		1 / (exp + 1),
		-1.0 / 2.0 * exp / expP2,
		1.0 / 6.0 * exp * (exp - 1) / expP3,
		-1.0 / 24.0 * exp * (-4*exp + exp2 + 1) / expP4,
		1.0 / 120.0 * exp * (11*exp - 11*exp2 + exp3 - 1) / expP5,
		-1.0 / 720.0 * exp * (-26*exp + 66*exp2 - 26*exp3 + exp4 + 1) / expP6,
		1.0 / 5040.0 * exp * (57*exp - 302*exp2 + 302*exp3 - 57*exp4 + exp5 - 1) / expP7,
		-1.0 / 40320.0 * exp * (-120*exp + 1191*exp2 - 2416*exp3 + 1191*exp4 - 120*exp5 + exp6 + 1) / expP8,
	}
}

func (p polynomial) Evaluate(x float32) float32 {
	coeff := float32(1)
	res := float32(0)
	for _, c := range p {
		res += c * coeff
		coeff *= x
	}
	return res
}

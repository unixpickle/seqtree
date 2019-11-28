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

type polynomial []float32

// newPolynomialLogSigmoid approximates the log of the
// sigmoid function as a high-order polynomial.
// The resulting polynomial is p(a) ~= log(sigmoid(x+a)).
// It is very accurate when abs(a) < 1.
func newPolynomialLogSigmoid(x float32) polynomial {
	// Create a short Taylor series.
	exp := math.Exp(float64(x))
	invExp := math.Exp(float64(-x))
	if math.IsInf(exp, 1) {
		// Use a big number that is right in the middle of
		// the range of 64-bit integers.
		exp = 1 << 32
	} else if math.IsInf(invExp, 1) {
		invExp = 1 << 32
	}

	logValue := float64(x)
	if x > -22 {
		// In the typical case, the loss function cannot
		// be approximated by f(x)=x.
		logValue = math.Log(1 / (1 + invExp))
	}

	exp2 := exp * exp
	exp3 := exp2 * exp
	exp4 := exp2 * exp2
	exp5 := exp3 * exp2
	exp6 := exp3 * exp3
	exp7 := exp4 * exp3

	expP := exp + 1
	expP2 := expP * expP
	expP3 := expP2 * expP
	expP4 := expP2 * expP2
	expP5 := expP4 * expP
	expP6 := expP3 * expP3
	expP7 := expP4 * expP3
	expP8 := expP4 * expP4
	expP9 := expP5 * expP4

	// Coefficients match the "Triangle of Eulerian numbers"
	// See https://oeis.org/A008292
	res64 := []float64{
		logValue,
		1 / (exp + 1),
		-1.0 / 2.0 * exp / expP2,
		1.0 / 6.0 * exp * (exp - 1) / expP3,
		-1.0 / 24.0 * exp * (-4*exp + exp2 + 1) / expP4,
		1.0 / 120.0 * exp * (11*exp - 11*exp2 + exp3 - 1) / expP5,
		-1.0 / 720.0 * exp * (-26*exp + 66*exp2 - 26*exp3 + exp4 + 1) / expP6,
		1.0 / 5040.0 * exp * (57*exp - 302*exp2 + 302*exp3 - 57*exp4 + exp5 - 1) / expP7,
		-1.0 / 40320.0 * exp * (-120*exp + 1191*exp2 - 2416*exp3 + 1191*exp4 - 120*exp5 + exp6 + 1) / expP8,
		1.0 / 362880.0 * exp * (247*exp - 4293*exp2 + 15619*exp3 - 15619*exp4 + 4293*exp5 - 247*exp6 + exp7 - 1) / expP9,
	}
	res := make(polynomial, len(res64))
	for i, x := range res64 {
		if math.IsNaN(x) {
			// If it's NaN, it's probably because x is so
			// far towards one of he extremes that a power
			// of exp is infinity. In this case, it's safe
			// to just treat the term as approximately 0.
			continue
		}
		res[i] = float32(x)
	}
	return res
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

type hessianMatrix struct {
	Dim    int
	Values []float32
}

func newHessianMatrixSoftmax(outputs, targets []float32) *hessianMatrix {
	max := outputs[0]
	for _, x := range outputs[1:] {
		if x > max {
			max = x
		}
	}

	exps := make([]float32, len(outputs))
	expSum := float32(0)
	for i, x := range outputs {
		exps[i] = float32(math.Exp(float64(x - max)))
		expSum += exps[i]
	}
	expSumSq := expSum * expSum

	targetSum := float32(0)
	for _, x := range targets {
		targetSum += x
	}

	res := &hessianMatrix{
		Dim:    len(outputs),
		Values: make([]float32, len(outputs)*len(outputs)),
	}

	for i := range outputs {
		for j := range outputs {
			val := -exps[j] * exps[i] / expSumSq
			if i == j {
				val += exps[i] / expSum
			}
			val *= targetSum
			res.Values[i+j*res.Dim] = val
		}
	}

	return res
}

func (h *hessianMatrix) Apply(v []float32) []float32 {
	if len(v) != h.Dim {
		panic("dimension mismatch")
	}
	res := make([]float32, h.Dim)
	for i := 0; i < h.Dim; i++ {
		rowIdx := i * h.Dim
		for j := 0; j < h.Dim; j++ {
			res[i] += h.Values[rowIdx+j] * v[j]
		}
	}
	return res
}

// ApplyInverse solves the equation Hx = v for x.
func (h *hessianMatrix) ApplyInverse(v []float32) []float32 {
	x := make([]float32, len(v))

	// Perform h.Dim steps of gradient descent.
	// In the future, this should probably use CG or some
	// more efficient method.
	for i := 0; i < h.Dim; i++ {
		residual := vectorDifference(v, h.Apply(x))
		product := h.Apply(residual)
		divisor := vectorNormSquared(product)
		if divisor == 0 {
			break
		}
		stepSize := vectorDot(residual, product) / divisor
		for j, y := range residual {
			x[j] += stepSize * y
		}
	}

	return x
}

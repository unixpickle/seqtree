package seqtree

import "sort"

// A Heuristic is a splitting critereon that, in some way
// or another, assesses how good a feature split is.
type Heuristic interface {
	// SampleVector generates a heuristic-specific vector
	// for the given sample. This vector is added together
	// with other sample vectors and passed to other
	// methods of the heuristic.
	SampleVector(sample *TimestepSample) []float32

	// Quality returns a scalar value indicating how much
	// improvement is likely to be made on the samples if
	// they are together in a leaf node.
	// Higher values mean more improvement, and zero means
	// no improvement.
	//
	// This value should scale with the number of samples.
	// For example, if leaf X contains a set of samples,
	// and leaf Y contains the samples of leaf X repeated
	// twice, then the quality of leaf Y should be roughly
	// twice that of leaf X.
	Quality(vectorSum []float32) float32

	// LeafOutput takes aggregate statistics about the
	// samples in a leaf node, and produces an output
	// delta for the leaf that should decrease the loss as
	// much as possible.
	LeafOutput(vectorSum []float32) []float32
}

// GradientHeuristic is a Heuristic which implements basic
// gradient boosting. Outputs are based on gradients, and
// quality is based on fitting the functional gradient in
// terms of mean squared error.
type GradientHeuristic struct {
	Loss GradLossFunc
}

func (g GradientHeuristic) SampleVector(sample *TimestepSample) []float32 {
	ts := sample.Timestep()
	return append([]float32{1}, g.Loss.LossGrad(ts.Output, ts.Target)...)
}

func (g GradientHeuristic) Quality(gradSum []float32) float32 {
	count := gradSum[0]
	if count == 0 {
		return 0
	}
	return vectorNormSquared(gradSum[1:]) / count
}

func (g GradientHeuristic) LeafOutput(gradSum []float32) []float32 {
	count := gradSum[0]
	res := make([]float32, len(gradSum)-1)
	s := -1 / count
	for i, x := range gradSum[1:] {
		res[i] = x * s
	}
	return res
}

// HessianHeuristic is a Heuristic which minimizes a
// quadratic approximation of the loss function for every
// leaf node.
type HessianHeuristic struct {
	Loss HessianLossFunc

	// Damping is the L2 penalty applied to the output
	// delta.
	//
	// It should typically be a small positive value
	// (e.g. 0.1) to prevent the inverse Hessian from
	// being too large in any given direction.
	Damping float32
}

func (h HessianHeuristic) SampleVector(sample *TimestepSample) []float32 {
	ts := sample.Timestep()
	grad := Softmax{}.LossGrad(ts.Output, ts.Target)
	hess := h.Loss.LossHessian(ts.Output, ts.Target)
	for i := 0; i < hess.Dim; i++ {
		hess.Data[i+i*hess.Dim] += h.Damping
	}
	return append(grad, hess.Data...)
}

func (h HessianHeuristic) Quality(sum []float32) float32 {
	_, v := h.minimize(sum)
	return -v
}

func (h HessianHeuristic) LeafOutput(sum []float32) []float32 {
	v, _ := h.minimize(sum)
	return v
}

func (h HessianHeuristic) minimize(sum []float32) ([]float32, float32) {
	dim := h.inferDimension(len(sum))
	grad := sum[:dim]
	hessian := &Hessian{
		Dim:  dim,
		Data: sum[dim:],
	}
	negGrad := make([]float32, len(grad))
	for i, x := range grad {
		negGrad[i] = -x
	}
	solution := hessian.ApplyInverse(negGrad)
	value := vectorDot(grad, solution) + 0.5*vectorDot(solution, hessian.Apply(solution))
	return solution, value
}

func (h HessianHeuristic) inferDimension(vecSize int) int {
	x := sort.Search(vecSize, func(n int) bool {
		return n*(n+1) >= vecSize
	})
	if x*(x+1) != vecSize {
		panic("invalid vector size")
	}
	return x
}

// A PolynomialHeuristic approximates the (binary) loss
// function as a high-order polynomial and minimizes it
// exactly for each split.
type PolynomialHeuristic struct{}

func (p PolynomialHeuristic) SampleVector(sample *TimestepSample) []float32 {
	ts := sample.Timestep()
	if len(ts.Output) != 2 {
		panic("PolynomialHeuristic is only supported with binary outputs")
	}
	poly := newPolynomialLogSigmoid(ts.Output[0] - ts.Output[1])
	poly1 := newPolynomialLogSigmoid(ts.Output[1] - ts.Output[0])
	for i, x := range poly {
		poly[i] = -x * ts.Target[0]
		if i%2 == 1 {
			poly[i] += poly1[i] * ts.Target[1]
		} else {
			poly[i] -= poly1[i] * ts.Target[1]
		}
	}

	// Getting rid of the constant term improves
	// numerical accuracy without changing the
	// objective
	poly[0] = 0

	return poly
}

func (p PolynomialHeuristic) Quality(sum []float32) float32 {
	_, y := p.minimize(sum)
	return -y
}

func (p PolynomialHeuristic) LeafOutput(sum []float32) []float32 {
	x, _ := p.minimize(sum)
	return []float32{x, -x}
}

func (p PolynomialHeuristic) minimize(poly polynomial) (float32, float32) {
	x := minimizeUnary(-1, 1, 30, poly.Evaluate)
	y := poly.Evaluate(x)
	return x, y
}

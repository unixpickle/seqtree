package seqtree

import (
	"math"
	"math/rand"
)

type LossFunc interface {
	Loss(outputs, targets []float32) float32
}

type GradLossFunc interface {
	LossFunc
	LossGrad(outputs, targets []float32) []float32
}

type HessianLossFunc interface {
	GradLossFunc
	LossHessian(outputs, targets []float32) *Hessian
}

type Softmax struct{}

// Sample samples a softmax distribution from logits.
func (s Softmax) Sample(outputs []float32) int {
	logProbs := s.logSoftmax(outputs)
	p := rand.Float32()
	for i, x := range logProbs {
		p -= float32(math.Exp(float64(x)))
		if p <= 0 {
			return i
		}
	}
	return len(outputs) - 1
}

// Loss computes the loss function given output logits and
// target probabilities.
func (s Softmax) Loss(outputs, targets []float32) float32 {
	// Code duplication from logSoftmax() prevents
	// allocations and can speedup computation
	// significantly.

	max := outputs[0]
	for _, x := range outputs[1:] {
		if x > max {
			max = x
		}
	}

	var sumOfExp float64
	for _, x := range outputs {
		sumOfExp += math.Exp(float64(x - max))
	}

	subtractor := max + float32(math.Log(float64(sumOfExp)))

	var loss float32
	for i, x := range outputs {
		loss += (subtractor - x) * targets[i]
	}
	return loss
}

// LossGrad computes the gradient of the softmax loss with
// respect to the outputs.
func (s Softmax) LossGrad(outputs, targets []float32) []float32 {
	var targetSum float32
	for _, x := range targets {
		targetSum += x
	}

	maxOutput := outputs[0]
	for _, x := range outputs[1:] {
		if x > maxOutput {
			maxOutput = x
		}
	}

	grad := make([]float32, len(outputs))
	for i, x := range outputs {
		grad[i] = float32(math.Exp(float64(x - maxOutput)))
	}
	var gradSum float32
	for _, x := range grad {
		gradSum += x
	}
	div := 1.0 / gradSum
	for i, x := range grad {
		grad[i] = targetSum*x*div - targets[i]
	}
	return grad
}

func (s Softmax) LossHessian(outputs, targets []float32) *Hessian {
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

	res := &Hessian{
		Dim:  len(outputs),
		Data: make([]float32, len(outputs)*len(outputs)),
	}

	for i := range outputs {
		for j := range outputs {
			val := -exps[j] * exps[i] / expSumSq
			if i == j {
				val += exps[i] / expSum
			}
			val *= targetSum
			res.Data[i+j*res.Dim] = val
		}
	}

	return res
}

func (s Softmax) logSoftmax(logits []float32) []float32 {
	max := logits[0]
	for _, x := range logits[1:] {
		if x > max {
			max = x
		}
	}

	var sumOfExp float64
	for _, x := range logits {
		sumOfExp += math.Exp(float64(x - max))
	}

	subtractor := max + float32(math.Log(float64(sumOfExp)))

	res := make([]float32, len(logits))
	for i, x := range logits {
		res[i] = x - subtractor
	}

	return res
}

// Hessian is a square symmetric matrix representing the
// second derivatives of a loss function.
type Hessian struct {
	Dim  int
	Data []float32
}

// Apply multiplies the Hessian by the vector v.
func (h *Hessian) Apply(v []float32) []float32 {
	if len(v) != h.Dim {
		panic("dimension mismatch")
	}
	res := make([]float32, h.Dim)
	for i := 0; i < h.Dim; i++ {
		rowIdx := i * h.Dim
		for j := 0; j < h.Dim; j++ {
			res[i] += h.Data[rowIdx+j] * v[j]
		}
	}
	return res
}

// ApplyInverse solves the equation Hx = v for x.
func (h *Hessian) ApplyInverse(v []float32) []float32 {
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

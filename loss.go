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

type PolynomialLossFunc interface {
	LossFunc
	LossPolynomialSize() int
	LossPolynomials(outputs, targets []float32) []Polynomial
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

// LossHessian computes the second derivatives of the
// logits with respect to the targets.
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

// MultiSoftmax is a softmax loss where each output
// contains multiple distinct softmax decisions.
type MultiSoftmax struct {
	// Sizes is the number of options for each softmax.
	Sizes []int
}

// Sample samples the softmax distributions from logits.
func (m MultiSoftmax) Sample(outputs []float32) []int {
	var samples []int
	for _, size := range m.Sizes {
		samples = append(samples, Softmax{}.Sample(outputs[:size]))
		outputs = outputs[size:]
	}
	if len(outputs) != 0 {
		panic("incorrect input size")
	}
	return samples
}

// Loss computes the loss function given output logits and
// target probabilities.
//
// The resulting loss is a sum of all the softmax losses.
func (m MultiSoftmax) Loss(outputs, targets []float32) float32 {
	var total float32
	for _, size := range m.Sizes {
		total += Softmax{}.Loss(outputs[:size], targets[:size])
		outputs = outputs[size:]
		targets = targets[size:]
	}
	if len(outputs) != 0 {
		println(len(outputs))
		panic("incorrect input size")
	}
	return total
}

// LossGrad computes the gradient of the softmax losses
// with respect to the outputs.
func (m MultiSoftmax) LossGrad(outputs, targets []float32) []float32 {
	var grad []float32
	for _, size := range m.Sizes {
		grad = append(grad, Softmax{}.LossGrad(outputs[:size], targets[:size])...)
		outputs = outputs[size:]
		targets = targets[size:]
	}
	return grad
}

type Sigmoid struct{}

// Sample samples a the logistic distribution.
func (s Sigmoid) Sample(outputs []float32) []bool {
	res := make([]bool, len(outputs))
	for i, x := range outputs {
		prob := 1 / (1 + math.Exp(float64(-x)))
		res[i] = rand.Float64() < prob
	}
	return res
}

// Loss computes the loss function given output logits and
// target probabilities.
func (s Sigmoid) Loss(outputs, targets []float32) float32 {
	var total float32
	for i, x := range outputs {
		t := targets[i]
		total += Softmax{}.Loss([]float32{x, 0}, []float32{t, 1 - t})
	}
	return total
}

// LossGrad computes the gradient of the sigmoid loss with
// respect to the logits.
func (s Sigmoid) LossGrad(outputs, targets []float32) []float32 {
	res := make([]float32, len(outputs))
	for i, x := range outputs {
		t := targets[i]
		g := Softmax{}.LossGrad([]float32{x, 0}, []float32{t, 1 - t})[0]
		res[i] = g
	}
	return res
}

func (s Sigmoid) LossHessian(outputs, targets []float32) *Hessian {
	res := &Hessian{
		Dim:  len(outputs),
		Data: make([]float32, len(outputs)*len(outputs)),
	}
	for i, x := range outputs {
		t := targets[i]
		h := Softmax{}.LossHessian([]float32{x, 0}, []float32{t, 1 - t})
		res.Data[i+i*len(outputs)] = h.Data[0]
	}
	return res
}

func (s Sigmoid) LossPolynomialSize() int {
	return 10
}

func (s Sigmoid) LossPolynomials(outputs, targets []float32) []Polynomial {
	res := make([]Polynomial, len(outputs))
	for i, x := range outputs {
		t := targets[i]
		p1 := newPolynomialLogSigmoid(x).Scale(t)
		p2 := newPolynomialLogSigmoid(-x).FlipX().Scale(1 - t)
		res[i] = p1.Add(p2).Scale(-1)
	}
	return res
}

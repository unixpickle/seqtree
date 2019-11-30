package main

import (
	"math/rand"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/seqtree"
)

type SequenceModel struct {
	Model *seqtree.Model
}

func NewSequenceModel() *SequenceModel {
	return &SequenceModel{
		Model: &seqtree.Model{BaseFeatures: SequenceLength + ImageSize*2},
	}
}

func (s *SequenceModel) Timesteps(samples []mnist.Sample) []*seqtree.Timestep {
	res := make([]*seqtree.Timestep, len(samples))
	for i, x := range samples {
		res[i] = sampleTimestep(NewBoolImgSample(x), rand.Intn(ImageSize), rand.Intn(ImageSize))
	}
	return res
}

func (s *SequenceModel) Sample() []bool {
	sample := NewBoolImg()
	for y := 0; y < ImageSize; y++ {
		for x := 0; x < ImageSize; x++ {
			ts := sampleTimestep(sample, x, y)
			seq := seqtree.Sequence{ts}
			s.Model.Evaluate(seq)
			sample.Set(x, y, seqtree.Sigmoid{}.Sample(ts.Output)[0])
		}
	}
	return sample
}

func (s *SequenceModel) AddTree(samples, validation []*seqtree.Timestep) (loss, delta float32) {
	shrinkage := float32(0.1)
	if len(s.Model.Trees) == 0 {
		// Take a large initial step.
		shrinkage = 1
	}

	seqs := make([]seqtree.Sequence, len(samples))
	validationSeqs := make([]seqtree.Sequence, len(validation))
	for i, sample := range samples {
		seqs[i] = seqtree.Sequence{sample}
	}
	for i, sample := range validation {
		validationSeqs[i] = seqtree.Sequence{sample}
	}
	s.Model.EvaluateAll(seqs)
	s.Model.EvaluateAll(validationSeqs)

	for _, seq := range seqs {
		loss += seq.MeanLoss(seqtree.Sigmoid{})
	}

	builder := seqtree.Builder{
		Heuristic: seqtree.PolynomialHeuristic{
			Loss: seqtree.Sigmoid{},
		},
		Depth:           5,
		MinSplitSamples: 100,
		Horizons:        []int{0},
		MaxUnion:        5,
	}
	pruner := seqtree.Pruner{
		Heuristic: builder.Heuristic,
		MaxLeaves: 10,
	}
	tree := builder.Build(seqtree.TimestepSamples(seqs))
	tree = pruner.Prune(seqtree.TimestepSamples(seqs), tree)
	seqtree.ScaleOptimalStep(seqtree.TimestepSamples(seqs), tree, seqtree.Sigmoid{},
		40.0, 10, 30)
	delta += seqtree.AvgLossDelta(seqtree.TimestepSamples(validationSeqs), tree, seqtree.Sigmoid{},
		shrinkage)
	s.Model.Add(tree, shrinkage)

	loss /= float32(len(samples))
	return
}

func (s *SequenceModel) MeanLoss(samples []*seqtree.Timestep) float32 {
	seqs := make([]seqtree.Sequence, len(samples))
	for i, sample := range samples {
		seqs[i] = seqtree.Sequence{sample}
	}
	s.Model.EvaluateAll(seqs)

	var loss float32
	for _, seq := range seqs {
		loss += seq.MeanLoss(seqtree.Sigmoid{})
	}
	return loss / float32(len(samples))
}

func sampleTimestep(img BoolImg, x, y int) *seqtree.Timestep {
	ts := &seqtree.Timestep{
		Features: seqtree.NewBitmap(SequenceLength + ImageSize*2),
		Output:   make([]float32, 1),
		Target:   make([]float32, 1),
	}
	seq := img.Window(x, y)
	for i, s := range seq {
		ts.Features.Set(i, s)
	}
	ts.Features.Set(SequenceLength+x, true)
	ts.Features.Set(SequenceLength+ImageSize+y, true)
	if img.At(x, y) {
		ts.Target[0] = 1
	}
	return ts
}

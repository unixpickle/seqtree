package main

import (
	"encoding/json"
	"io/ioutil"
	"os"

	"github.com/pkg/errors"
	"github.com/unixpickle/seqtree"
)

const (
	ImageSize      = 28
	SequenceLength = ImageSize * ImageSize
)

type SequenceModel struct {
	Models []*seqtree.Model
}

func NewSequenceModel() *SequenceModel {
	res := &SequenceModel{}
	for i := 0; i < SequenceLength; i++ {
		res.Models = append(res.Models, &seqtree.Model{
			BaseFeatures: i,
		})
	}
	return res
}

func (s *SequenceModel) Save(path string) error {
	data, err := json.Marshal(s)
	if err != nil {
		return errors.Wrap(err, "save model")
	}
	if err := ioutil.WriteFile(path, data, 0755); err != nil {
		return errors.Wrap(err, "save model")
	}
	return nil
}

func (s *SequenceModel) Load(path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return errors.Wrap(err, "load model")
	}
	if err := json.Unmarshal(data, s); err != nil {
		return errors.Wrap(err, "load model")
	}
	return nil
}

func (s *SequenceModel) NumTrees() int {
	return len(s.Models[1].Trees)
}

func (s *SequenceModel) Sample() []bool {
	var sample []bool
	for _, model := range s.Models {
		ts := s.sampleTimestep(model, sample)
		model.Evaluate(seqtree.Sequence{ts})
		value := seqtree.Sigmoid{}.Sample(ts.Output)[0]
		sample = append(sample, value)
	}
	return sample
}

func (s *SequenceModel) AddTree(boolSeqs [][]bool) (loss, delta float32) {
	const shrinkage = 0.1
	for _, model := range s.Models {
		seqs := make([]seqtree.Sequence, len(boolSeqs))
		for i, boolSeq := range boolSeqs {
			seqs[i] = seqtree.Sequence{s.sampleTimestep(model, boolSeq)}
		}
		model.EvaluateAll(seqs)

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
		delta += seqtree.AvgLossDelta(seqtree.TimestepSamples(seqs), tree, seqtree.Sigmoid{},
			shrinkage)
		model.Add(tree, shrinkage)
	}
	loss /= float32(len(boolSeqs))
	return
}

func (s *SequenceModel) MeanLoss(boolSeqs [][]bool) float32 {
	var loss float32
	for _, model := range s.Models {
		seqs := make([]seqtree.Sequence, len(boolSeqs))
		for i, boolSeq := range boolSeqs {
			seqs[i] = seqtree.Sequence{s.sampleTimestep(model, boolSeq)}
		}
		model.EvaluateAll(seqs)

		for _, seq := range seqs {
			loss += seq.MeanLoss(seqtree.Sigmoid{})
		}
	}
	return loss / float32(len(boolSeqs))
}

func (s *SequenceModel) sampleTimestep(model *seqtree.Model, seq []bool) *seqtree.Timestep {
	numPrefix := model.BaseFeatures
	ts := &seqtree.Timestep{
		Features: seqtree.NewBitmap(model.NumFeatures()),
		Output:   make([]float32, 1),
		Target:   make([]float32, 1),
	}
	for i, s := range seq[:numPrefix] {
		if s {
			ts.Features.Set(i, true)
		}
	}
	if len(seq) > numPrefix {
		if seq[numPrefix] {
			ts.Target[0] = 1
		}
	}
	return ts
}

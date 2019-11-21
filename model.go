package seqtree

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"runtime"
	"sync"

	"github.com/gonum/blas/blas32"
	"github.com/pkg/errors"
)

// A Model is a sequence prediction model.
type Model struct {
	// BaseFeatures is the number of features that come
	// with the data by default.
	BaseFeatures int

	// ExtraFeatures is the number of features that this
	// model adds to the data.
	ExtraFeatures int

	// Trees is an ensemble of trees comprising the model.
	// This slice is ordered, and trees should be run from
	// first to last.
	Trees []*Tree
}

// NumFeatures gets the total number of features expected
// in sequences by this model.
func (m *Model) NumFeatures() int {
	return m.BaseFeatures + m.ExtraFeatures
}

// Evaluate evaluates the model on the sequence.
// At the end of the evaluation, all of the features and
// output vectors in the sequence will be updated.
func (m *Model) Evaluate(seq Sequence) {
	m.EvaluateAt(seq, 0)
}

// EvaluateAt is like Evaluate(), but it starts at the
// given index of the sequence.
func (m *Model) EvaluateAt(seq Sequence, start int) {
	for _, t := range m.Trees {
		for i, ts := range seq[start:] {
			leaf := t.Evaluate(&TimestepSample{Sequence: seq, Index: i + start})
			v1 := blas32.Vector{Inc: 1, Data: leaf.OutputDelta}
			v2 := blas32.Vector{Inc: 1, Data: ts.Output}
			blas32.Axpy(len(ts.Output), 1.0, v1, v2)
			if leaf.Feature != 0 {
				ts.Features.Set(leaf.Feature, true)
			}
		}
	}
}

// EvaluateAll evaluates the model on a list of sequences.
func (m *Model) EvaluateAll(seqs []Sequence) {
	ch := make(chan Sequence, len(seqs))
	for _, x := range seqs {
		ch <- x
	}
	close(ch)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for seq := range ch {
				m.Evaluate(seq)
			}
		}()
	}

	wg.Wait()
}

// Add adds a tree to the model, scaling it according to
// the negative of stepSize.
func (m *Model) Add(t *Tree, stepSize float32) {
	t.Scale(-stepSize)
	m.Trees = append(m.Trees, t)
	m.ExtraFeatures += t.NumFeatures()
}

// Save saves the model to a JSON file.
func (m *Model) Save(path string) error {
	data, err := json.Marshal(m)
	if err != nil {
		return errors.Wrap(err, "save model")
	}
	if err := ioutil.WriteFile(path, data, 0755); err != nil {
		return errors.Wrap(err, "save model")
	}
	return nil
}

// Load loads the model from a JSON file.
// Does not fail with an error if the file does not exist.
func (m *Model) Load(path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return errors.Wrap(err, "load model")
	}
	if err := json.Unmarshal(data, m); err != nil {
		return errors.Wrap(err, "load model")
	}
	return nil
}

// Tree represents part of a decision tree.
// Leaf nodes have a non-nil leaf, and branches have a
// non-nil branch.
type Tree struct {
	Branch *Branch
	Leaf   *Leaf
}

// Evaluate runs the tree for the timestep.
func (t *Tree) Evaluate(ts *TimestepSample) *Leaf {
	if t.Leaf != nil {
		return t.Leaf
	}
	for _, f := range t.Branch.Feature {
		if ts.BranchFeature(f) {
			return t.Branch.TrueBranch.Evaluate(ts)
		}
	}
	return t.Branch.FalseBranch.Evaluate(ts)
}

// NumFeatures gets the number of new features added by
// the tree.
func (t *Tree) NumFeatures() int {
	if t.Leaf != nil {
		if t.Leaf.Feature != 0 {
			return 1
		} else {
			return 0
		}
	} else {
		return t.Branch.FalseBranch.NumFeatures() + t.Branch.TrueBranch.NumFeatures()
	}
}

// Scale scales the tree.
func (t *Tree) Scale(s float32) {
	if t.Leaf != nil {
		for i, x := range t.Leaf.OutputDelta {
			t.Leaf.OutputDelta[i] = x * s
		}
	} else {
		t.Branch.FalseBranch.Scale(s)
		t.Branch.TrueBranch.Scale(s)
	}
}

// Branch represents tree nodes that split into two
// sub-nodes.
type Branch struct {
	Feature     BranchFeatureUnion
	FalseBranch *Tree
	TrueBranch  *Tree
}

// Leaf represents terminal tree nodes.
type Leaf struct {
	// OutputDelta is the vector to add to the prediction
	// outputs at the current timestep.
	OutputDelta []float32

	// If non-zero, this is a feature to set in the sample
	// at the current timestep, in addition to the
	// prediction.
	Feature int
}

// A BranchFeature is a feature identifier to look at in a
// given branch of a tree.
type BranchFeature struct {
	// The feature index to split on.
	// Can be -1 to indicate that StepsInPast goes beyond
	// the beginning of the sequence.
	Feature int

	// StepsInPast indicates how many timesteps ago we
	// will look at the feature. A value of 0 means the
	// feature at the most recent timestep.
	StepsInPast int
}

// A BranchFeatureUnion is a logical OR of BranchFeatures.
// An empty union is always false.
type BranchFeatureUnion []BranchFeature

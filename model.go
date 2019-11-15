package seqtree

import (
	"runtime"
	"sync"

	"github.com/gonum/blas/blas32"
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
func (m *Model) Evaluate(start *Timestep) {
	for _, t := range m.Trees {
		start.Iterate(func(ts *Timestep) {
			leaf := t.Evaluate(ts)
			v1 := blas32.Vector{Inc: 1, Data: leaf.OutputDelta}
			v2 := blas32.Vector{Inc: 1, Data: ts.Output}
			blas32.Axpy(len(ts.Output), 1.0, v1, v2)
			if leaf.Feature != 0 {
				ts.Features[leaf.Feature] = true
			}
		})
	}
}

// EvaluateAll evaluates the model on a list of sequences.
func (m *Model) EvaluateAll(starts []*Timestep) {
	ch := make(chan *Timestep, len(starts))
	for _, x := range starts {
		ch <- x
	}
	close(ch)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ts := range ch {
				m.Evaluate(ts)
			}
		}()
	}
}

// Add adds a tree to the model, scaling it according to
// the negative of stepSize.
func (m *Model) Add(t *Tree, stepSize float32) {
	t.Scale(-stepSize)
	m.Trees = append(m.Trees, t)
	m.ExtraFeatures += t.NumFeatures()
}

// Tree represents part of a decision tree.
// Leaf nodes have a non-nil leaf, and branches have a
// non-nil branch.
type Tree struct {
	Branch *Branch
	Leaf   *Leaf
}

// Evaluate runs the tree for the timestep.
func (t *Tree) Evaluate(ts *Timestep) *Leaf {
	if t.Leaf != nil {
		return t.Leaf
	}
	if ts.BranchFeature(t.Branch.Feature) {
		return t.Branch.TrueBranch.Evaluate(ts)
	} else {
		return t.Branch.FalseBranch.Evaluate(ts)
	}
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
	Feature     BranchFeature
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

package seqtree

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

// Evaluate evaluates the model on the sequence.
// At the end of the evaluation, all of the features and
// output vectors in the sequence will be updated.
func (m *Model) Evaluate(start *Timestep) {
	for _, t := range m.Trees {
		start.Iterate(func(ts *Timestep) {
			leaf := t.Evaluate(ts)
			for i, x := range leaf.OutputDelta {
				ts.Output[i] += x
			}
			if leaf.Feature != 0 {
				ts.Features[leaf.Feature] = true
			}
		})
	}
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
	OutputDelta []float64

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

package seqtree

import "math"

// A Pruner stores parameters for pruning trees to prevent
// overfitting.
type Pruner struct {
	// Heuristic is used to determine the quality of
	// splits.
	Heuristic Heuristic

	// MaxLeaves is the maximum number of leaves for
	// pruned trees to have.
	MaxLeaves int
}

func (p *Pruner) Prune(samples []*TimestepSample, t *Tree) {
	if p.MaxLeaves < 1 {
		panic("cannot restrict to fewer than 1 leaves")
	}

	vecSamples := newVecSamples(p.Heuristic, samples)
	sums := p.leafSums(vecSamples, t)
	bases := p.baseBranches(t)

	for len(sums) > p.MaxLeaves {
		lowestScore := float32(math.Inf(1))
		var lowestBase *Tree
		for _, base := range bases {
			sum1 := sums[base.Branch.FalseBranch.Leaf].Sum()
			sum2 := sums[base.Branch.TrueBranch.Leaf].Sum()
			combinedSum := newKahanSum(len(sum1))
			combinedSum.Add(sum1)
			combinedSum.Add(sum2)
			score := p.Heuristic.Quality(sum1) + p.Heuristic.Quality(sum2) -
				p.Heuristic.Quality(combinedSum.Sum())
			if score < lowestScore {
				lowestScore = score
				lowestBase = base
			}
		}
		b := lowestBase.Branch
		newSum := sums[b.FalseBranch.Leaf]
		newSum.Add(sums[b.TrueBranch.Leaf].Sum())
		delete(sums, b.FalseBranch.Leaf)
		delete(sums, b.TrueBranch.Leaf)
		lowestBase.Branch = nil
		lowestBase.Leaf = &Leaf{
			OutputDelta: p.Heuristic.LeafOutput(newSum.Sum()),
		}
		sums[lowestBase.Leaf] = newSum
		bases = p.baseBranches(t)
	}
}

func (p *Pruner) baseBranches(t *Tree) []*Tree {
	if t.Leaf != nil {
		return nil
	} else if t.Branch.FalseBranch.Leaf != nil && t.Branch.TrueBranch.Leaf != nil {
		return []*Tree{t}
	} else {
		return append(p.baseBranches(t.Branch.FalseBranch),
			p.baseBranches(t.Branch.TrueBranch)...)
	}
}

func (p *Pruner) leafSums(samples []vecSample, t *Tree) map[*Leaf]*kahanSum {
	res := map[*Leaf]*kahanSum{}
	for _, s := range samples {
		leaf := t.Evaluate(&s.TimestepSample)
		if sum, ok := res[leaf]; ok {
			sum.Add(s.Vector)
		} else {
			sum = newKahanSum(len(s.Vector))
			sum.Add(s.Vector)
			res[leaf] = sum
		}
	}
	return res
}

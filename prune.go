package seqtree

import (
	"math"
	"runtime"
	"sync"

	"github.com/unixpickle/essentials"
)

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

func (p *Pruner) Prune(samples []*TimestepSample, t *Tree) *Tree {
	if p.MaxLeaves < 1 {
		panic("cannot restrict to fewer than 1 leaves")
	}
	vecSamples := newVecSamples(p.Heuristic, samples)
	result := t
	for len(result.Leaves()) > p.MaxLeaves {
		result = p.bestPrune(vecSamples, result)
	}
	if result != t {
		result = result.Copy()
		p.recomputeOutputDeltas(vecSamples, result)
	}
	return result
}

func (p *Pruner) bestPrune(samples []vecSample, t *Tree) *Tree {
	leaves := t.Leaves()

	var lock sync.Mutex
	var bestTree *Tree
	bestQuality := float32(math.Inf(-1))

	var wg sync.WaitGroup
	numProcs := essentials.MinInt(runtime.GOMAXPROCS(0), len(leaves))
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := i; j < len(leaves); j += numProcs {
				t1 := pruneLeaf(t, leaves[j])
				q := p.treeQuality(samples, t1)
				lock.Lock()
				if q > bestQuality {
					bestQuality = q
					bestTree = t1
				}
				lock.Unlock()
			}
		}(i)
	}
	wg.Wait()

	return bestTree
}

func (p *Pruner) treeQuality(samples []vecSample, t *Tree) float32 {
	sums := p.leafSums(samples, t)
	quality := newKahanSum(1)
	for _, s := range sums {
		quality.Add([]float32{p.Heuristic.Quality(s.Sum())})
	}
	return quality.Sum()[0]
}

func (p *Pruner) recomputeOutputDeltas(samples []vecSample, t *Tree) {
	sums := p.leafSums(samples, t)
	for l, s := range sums {
		l.OutputDelta = p.Heuristic.LeafOutput(s.Sum())
	}
}

func (p *Pruner) leafSums(samples []vecSample, t *Tree) map[*Leaf]*kahanSum {
	sums := map[*Leaf]*kahanSum{}
	for _, s := range samples {
		leaf := t.Evaluate(&s.TimestepSample)
		if sum, ok := sums[leaf]; ok {
			sum.Add(s.Vector)
		} else {
			sum = newKahanSum(len(s.Vector))
			sum.Add(s.Vector)
			sums[leaf] = sum
		}
	}
	return sums
}

func pruneLeaf(t *Tree, l *Leaf) *Tree {
	if t.Leaf == l {
		panic("cannot prune root")
	} else if t.Leaf != nil {
		return t
	} else if t.Branch.FalseBranch.Leaf == l {
		return t.Branch.TrueBranch
	} else if t.Branch.TrueBranch.Leaf == l {
		return t.Branch.FalseBranch
	} else {
		return &Tree{
			Branch: &Branch{
				Feature:     t.Branch.Feature,
				FalseBranch: pruneLeaf(t.Branch.FalseBranch, l),
				TrueBranch:  pruneLeaf(t.Branch.TrueBranch, l),
			},
		}
	}
}

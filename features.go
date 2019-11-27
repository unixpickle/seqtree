package seqtree

// AddLeafFeatures creates a new feature for each leaf in
// the tree, starting at the given index.
func AddLeafFeatures(t *Tree, start int) {
	for i, l := range t.Leaves() {
		l.Feature = start + i
	}
}

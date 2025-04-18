from genes import NodeGene, EdgeGene
from chromosome import Chromosome

C0 = Chromosome(
    id=0,
    nodes=[
        NodeGene(0, "input"),
        NodeGene(1, "input"),
        NodeGene(2, 1),
        NodeGene(3, 1),
        NodeGene(4, "output"),
        NodeGene(5, "output"),
    ],
    edges=[
        EdgeGene(0, 0, 2, 0.7),
        EdgeGene(1, 1, 2, 0.2),
        EdgeGene(2, 0, 3, 0.5),
        EdgeGene(3, 1, 3, 0.1),
        EdgeGene(4, 2, 4, 0.3),
        EdgeGene(5, 2, 5, 0.4),
        EdgeGene(6, 3, 4, 0.8),
        EdgeGene(7, 3, 5, 0.9),
    ],
)
C0.show(save=True)

C1 = Chromosome(
    id=1,
    nodes=[
        NodeGene(0, "input"),
        NodeGene(1, "input"),
        NodeGene(2, 1),
        NodeGene(3, "output"),
        NodeGene(4, "output"),
    ],
    edges=[
        EdgeGene(0, 0, 2, 0.85),
        EdgeGene(1, 1, 2, 0.1),
        EdgeGene(2, 2, 3, 0.4),
        EdgeGene(3, 2, 4, 1.0),
    ],
)
C1.show(save=True)

C2 = Chromosome(
    id=2,
    nodes=[
        NodeGene(0, "input"),
        NodeGene(1, "input"),
        NodeGene(2, 1),
        NodeGene(3, 1),
        NodeGene(4, 1),
        NodeGene(5, "output"),
        NodeGene(6, "output"),
    ],
    edges=[
        EdgeGene(0, 0, 2, 0.5),
        EdgeGene(1, 1, 3, -0.6),
        EdgeGene(2, 0, 4, 0.3),
        EdgeGene(3, 1, 2, -0.1),
        EdgeGene(4, 1, 4, 0.9),
        EdgeGene(5, 2, 5, 1.0),
        EdgeGene(6, 3, 5, 0.8),
        EdgeGene(7, 3, 6, -0.7),
        EdgeGene(8, 4, 6, 0.4),
        EdgeGene(9, 2, 6, 0.2),
    ],
)
C2.show(save=True)

C3 = Chromosome(id=3, inputs=2, hidden=3, outputs=3)
C3.show(save=True)

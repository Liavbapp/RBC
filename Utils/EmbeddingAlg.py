from karateclub import GLEE, Node2Vec, GraphWave, AE, BigClam, BoostNE, DANMF, DeepWalk, Diff2Vec, FeatherNode, GL2Vec, \
    GraRep, LaplacianEigenmaps, MNMF, MUSAE, NetMF, NMFADMM, NNSED, NodeSketch, RandNE, Role2Vec, SocioDim, SymmNMF, \
    Walklets

from Utils.CommonStr import EmbeddingAlgorithms


def get_embedding_algo(alg_name, dimensions, args=None):
    if alg_name == EmbeddingAlgorithms.glee:
        return GLEE(dimensions=dimensions - 1)
    if alg_name == EmbeddingAlgorithms.graph_wave:
        return GraphWave(sample_number=dimensions)
    if alg_name == EmbeddingAlgorithms.ae:
        return AE(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.big_clam:
        return BigClam(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.boost_ne:
        return BoostNE(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.danmf:
        raise NotImplementedError
    if alg_name == EmbeddingAlgorithms.deep_walk:
        return DeepWalk(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.feather_node:
        return FeatherNode(reduction_dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.gl2vec:
        return GL2Vec(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.gra_rep:
        return GraRep(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.laplacian_eigenmaps:
        return LaplacianEigenmaps(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.mnmf:
        return MNMF(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.musae:
        return MUSAE(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.net_mf:
        return NetMF(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.nmfadmm:
        return NMFADMM(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.nnsed:
        return NNSED(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.node_sketch:
        return NodeSketch(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.rand_ne:
        return RandNE(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.role2vec:
        return Role2Vec(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.socio_dim:
        return SocioDim(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.symm_nmf:
        return SymmNMF(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.walklets:
        return Walklets(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.diff2vec:
        if args is not None:
            return Diff2Vec(dimensions=dimensions, diffusion_number=args['diffusion_number'], diffusion_cover=args['diffusion_cover'])
        else:
            return Diff2Vec(dimensions=dimensions)
    if alg_name == EmbeddingAlgorithms.node2vec:
        if args is not None:
            return Node2Vec(dimensions=dimensions, p=args['p'], q=args['q'])
        else:
            return Node2Vec(dimensions=dimensions)

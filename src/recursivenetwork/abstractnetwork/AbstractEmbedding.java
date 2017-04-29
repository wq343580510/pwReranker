package recursivenetwork.abstractnetwork;

/** Interface that must be implemented by any computation graph that represents an embedding.
 * @author Dipendra Misra (dkm@cs.cornell.edu) */
public interface AbstractEmbedding {
	
	/** dimensionality of the embedding */
	public int getDimension();
	
	/** get embedding of an object*/
	public Object getEmbedding(Object obj);
}

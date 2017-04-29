package recursivenetwork.util;

/** A simple pair class 
 * 
 * @author Dipendra Misra (dkm@cs.cornell.edu) */
public class Pair<E, F> {

	private final E first;
	private final F second;
	
	private Pair(E first, F second) {
		this.first = first;
		this.second = second;
	}
	
	public static <E,F> Pair<E,F> of(E first, F second) {
		return new Pair<E,F>(first, second);
	}
	
	public E first() {
		return this.first;
	}
	
	public F second() {
		return this.second;
	}
}

/*!
 * This procedure provides local estimates of the intrinsic dimension 
 * around a query point. 
 * The implementation is valid for template instantiations both as
 * <code>T = float</code> and <code>T = double</code>
 *
 * @param Q List of neighbor distances.
 * @param n Rank of neighbor corresponding to the last entry in <code>Q</code>.
 * @return Local estimate of intrinsic dimension.
 */

template <typename T>
T evalID(	T* Q,
		const int n) {
	//cout << "inside evalID" <<endl;
	T w = Q[n-1];
	//cout << "w = "<<w<<endl;
	if (w == 0) return std::numeric_limits<T>::max();
	if (Q[0] == w) return std::numeric_limits<T>::max();
	T sum = 0;
	for (int i=0; i < n-1; i++) {
		//cout << "Q["<<i<<"] = "<<Q[i]<<endl;
		sum += log(Q[i] / w);
	}
	//cout << "sum = "<<sum<<endl;
	sum = sum / (n-1);
	//cout << "id = "<< (-1 / sum) <<endl;
	return -1 / sum;
}// end evalID

vector<int> observers(	const int nb,
			const int n) {
	vector<int> obs(nb);
	if (nb < n) {
		srand(time(NULL));
		// The observers' selection is not perfectly uniform.
		float resolution = float(n)/float(nb);
		float indBegin = 0;
		float indEnd   = resolution;
		for (int i=0 ; i<nb ; i++) {
			obs[i] = int(indBegin) + (rand() % (int(indEnd)-int(indBegin)));
			indBegin = indEnd;
			indEnd  += resolution;
		}
	}else{
		obs.resize(n);
		for (int i=0 ; i<n ; i++) {
			obs[i] = i;
		}
	}
	/*for (int i=0 ; i<nb ; i++) {
		cout << obs[i] << " ";
	}
	cout << endl;*/
	return obs;
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package linear_scan_search;

/**
 *
 * @author arwawali
 */
class knn_dis_pair implements Comparable<knn_dis_pair> {
    
    private int index;
    private float dis; 
    public knn_dis_pair(int i, float d){
        this.index=i;
        this.dis=d;
    }
    public float getDis(){
        return this.dis;
    }

    public int getIndex() {
        return index;
    }

    //@Override
    public int compareTo(knn_dis_pair o) {
         int compare=Float.compare(this.getDis(), o.getDis());
         return compare;
         /*if (this.getDis() >= o.getDis())
            return 1;
        else
            return 0; */
         
        
    }
}

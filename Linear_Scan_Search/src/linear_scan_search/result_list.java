/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package linear_scan_search;

import java.util.BitSet;

/**
 *
 * @author arwawali
 */
class result_list implements Comparable<result_list>{
        BitSet finger_prints;
        int objects_index;
        float dis;
        
        public result_list(int i, float d, BitSet f){
            this.objects_index=i;
            this.finger_prints=f;
            this.dis=d;
        }
        float getDis(){
            return this.dis;
        }
         //@Override
            public int compareTo(result_list o) {
                 int compare=Float.compare(this.getDis(), o.getDis());
                 return compare;
         /*if (this.getDis() >= o.getDis())
            return 1;
        else
            return 0; */
         
        
             }
        
        
    }

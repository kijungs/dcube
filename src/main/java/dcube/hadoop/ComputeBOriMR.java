/* =================================================================================
 *
 * D-Cube: Dense-Block Detection in Terabyte-Scale Tensors
 * Authors: Kijung Shin, Bryan Hooi, Jisu Kim, and Christos Faloutsos
 *
 * Version: 1.0
 * Date: August 6, 2016
 * Main Contact: Kijung Shin (kijungs@cs.cmu.edu)
 *
 * This software is free of charge under research purposes.
 * For commercial purposes, please contact the author.
 *
 * =================================================================================
 */

package dcube.hadoop;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.*;

/**
 * MR job for computing BOri
 * @author kijungs
 */
public class ComputeBOriMR {

    public static class ComputeBOriMapper extends Mapper<Object, Text, NullWritable, Text> {

        private int dimension = 0;
        private boolean[][] modeToAttValsIncluded;

        @Override
        public void setup(Context context
        ) throws IOException, InterruptedException {
            dimension = context.getConfiguration().getInt(Parameter.PARAM_DIMENSION, 0);
            modeToAttValsIncluded = new boolean[dimension][];
            for(int mode = 0; mode < dimension; mode++) {
                int length = context.getConfiguration().getInt(Parameter.PARAM_CARDINALITY + mode, 0);
                modeToAttValsIncluded[mode] = new boolean[length];
            }
            Path[] localPaths = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if(localPaths != null && localPaths.length > 0) {
                for(Path path : localPaths) {
                    ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path.toString()), 8388608));
                    try {
                        while (true) {
                            byte mode = in.readByte();
                            modeToAttValsIncluded[mode][in.readInt()] = true;
                        }
                    } catch(EOFException e) {
                    }
                    in.close();
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");

            boolean included = true;
            for(int mode = 0; mode < dimension; mode++) {
                if (!modeToAttValsIncluded[mode][Integer.valueOf(tokens[mode])]) {
                    included = false;
                    break;
                }
            }

            if(included) {
                context.write(NullWritable.get(), value);
            }
        }
    }

    public static class ComputeBOriReducer extends Reducer<NullWritable, Text, NullWritable, Text> {
        //do nothing
    }

}

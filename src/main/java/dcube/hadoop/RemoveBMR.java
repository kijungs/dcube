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
 * MR job for removing attribute values and the corresponding tuples from the current block B
 * @author kijungs
 */
public class RemoveBMR {

    public static class RemoveBMapper extends Mapper<Object, Text, NullWritable, Text> {

        private int modeToRemove;
        private boolean[] attValesToRemove;

        @Override
        public void setup(Mapper<Object, Text, NullWritable, Text>.Context context
        ) throws IOException, InterruptedException {
            modeToRemove = context.getConfiguration().getInt(Parameter.PARAM_MODE_TO_REMOVE, 0);
            int length = context.getConfiguration().getInt(Parameter.PARAM_CARDINALITY + modeToRemove, 0);
            attValesToRemove = new boolean[length];
            Path[] localPaths = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if(localPaths != null && localPaths.length > 0) {
                for(Path path : localPaths) {
                    ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path.toString()), 8388608));
                    try {
                        while (true) {
                            attValesToRemove[in.readInt()] = true;
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
            if(!attValesToRemove[Integer.valueOf(tokens[modeToRemove])]) {
                context.write(NullWritable.get(), value);
            }
        }
    }

    public static class RemoveBReducer extends Reducer<NullWritable, Text, NullWritable, Text> {
        //do nothing
    }

}

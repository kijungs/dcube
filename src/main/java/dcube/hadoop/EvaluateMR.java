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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.*;

/**
 * MR job for computing the mass of a found block
 * @author kijungs
 */
public class EvaluateMR {

    public static class EvaluateMapper extends Mapper<Object, Text, IntWritable, LongWritable> {

        private int dimension = 0;
        private boolean[][] modeToAttValsIncluded;
        private long mass = 0;

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

            boolean remove = true;
            for(int mode = 0; mode < dimension; mode++) {
                if (!modeToAttValsIncluded[mode][Integer.valueOf(tokens[mode])]) {
                    remove = false;
                    break;
                }
            }

            if(remove) {
                mass += Integer.valueOf(tokens[dimension]);
            }
        }

        @Override
        public void cleanup(Mapper<Object, Text, IntWritable, LongWritable>.Context context
        ) throws IOException, InterruptedException {
            LongWritable longWritable = new LongWritable(mass);
            context.write(new IntWritable(0), longWritable);
        }
    }

    public static class EvaluateReducer extends Reducer<IntWritable, LongWritable, NullWritable, Text> {

        @Override
        public void reduce(IntWritable key, Iterable<LongWritable> values,
                           Reducer<IntWritable, LongWritable, NullWritable, Text>.Context context
        ) throws IOException, InterruptedException {
            long mass = 0;
            for(LongWritable value : values) {
                mass += value.get();
            }
            context.write(NullWritable.get(), new Text(""+mass));
        }
    }

}

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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * MR job for computing the cardinality of each attribute
 * @author kijungs
 */
public class CardinalityMR {

    public static class CardinalityMapper
            extends Mapper<Object, Text, IntWritable, IntWritable> {

        private int dimension = 0;
        private int[] maxModeAttValues;

        @Override
        public void setup(Context context
        ) throws IOException, InterruptedException {
            dimension = context.getConfiguration().getInt(Parameter.PARAM_DIMENSION, 0);
            maxModeAttValues = new int[dimension];
        }

        @Override
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            for(int mode=0; mode< dimension; mode++) {
                int index = Integer.valueOf(tokens[mode]);
                maxModeAttValues[mode] = Math.max(index, maxModeAttValues[mode]);
            }
        }

        @Override
        public void cleanup(Context context
        ) throws IOException, InterruptedException {
            for(int mode=0; mode< dimension; mode++) {
                context.write(new IntWritable(mode), new IntWritable(maxModeAttValues[mode]));
            }
        }
    }

    public static class CardinalityReducer
            extends Reducer<IntWritable, IntWritable, NullWritable, Text> {

        final Text text = new Text("");

        @Override
        public void reduce(IntWritable key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int mode = key.get();
            int maxAttVal = 0;
            for (IntWritable value : values) {
                maxAttVal = Math.max(value.get(), maxAttVal);
            }
            text.set(mode + "," + (maxAttVal + 1));
            context.write(NullWritable.get(), text);
        }

    }
}

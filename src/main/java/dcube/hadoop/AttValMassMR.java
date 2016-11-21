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
 * MR job for computing attribute-value masses
 * @author kijungs
 */
public class AttValMassMR {


    public static class AttValMassMapper
            extends Mapper<Object, Text, IntWritable, IntWritable> {

        private int dimension = 0;
        private int[] cardinalities = null;
        private int[][] attValMasses = null;
        private int maxCardinality = 0;
        private IntWritable keyWritable = new IntWritable();
        private IntWritable valueWritable = new IntWritable();

        @Override
        public void setup(Context context
        ) throws IOException, InterruptedException {
            dimension = context.getConfiguration().getInt(Parameter.PARAM_DIMENSION, 0);
            cardinalities = new int[dimension];
            attValMasses = new int[dimension][];
            for(int mode = 0; mode < dimension; mode++) {
                int length = context.getConfiguration().getInt(Parameter.PARAM_CARDINALITY + mode, 0);
                cardinalities[mode] = length;
                maxCardinality = Math.max(maxCardinality, length);
                attValMasses[mode] = new int[cardinalities[mode]];
            }
        }

        @Override
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            int tupleValue = Integer.valueOf(tokens[dimension]);
            for(int mode=0; mode< dimension; mode++) {
                int index = Integer.valueOf(tokens[mode]);
                attValMasses[mode][index] += tupleValue;
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            for(int mode = 0; mode < dimension; mode++) {
                for(int index = 0; index < cardinalities[mode]; index++) {
                    if(attValMasses[mode][index] > 0) {
                        keyWritable.set(maxCardinality * mode + index);
                        valueWritable.set(attValMasses[mode][index]);
                        context.write(keyWritable, valueWritable);
                    }
                }
            }
        }
    }

    public static class AttValMassReducer
            extends Reducer<IntWritable, IntWritable, NullWritable, Text> {

        private Text text = new Text("");
        private int dimension = 0;
        private int maxCardinality = 0;

        @Override
        public void setup(Context context
        ) throws IOException, InterruptedException {
            dimension = context.getConfiguration().getInt(Parameter.PARAM_DIMENSION, 0);
            for(int mode = 0; mode < dimension; mode++) {
                int length = context.getConfiguration().getInt(Parameter.PARAM_CARDINALITY + mode, 0);
                maxCardinality = Math.max(maxCardinality, length);
            }
        }

        @Override
        public void reduce(IntWritable key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int mode = key.get()/ maxCardinality;
            int attValue = key.get() - mode * maxCardinality;
            long sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            text.set(mode + "," + attValue + "," + sum);
            context.write(NullWritable.get(), text);
        }
    }
}

import java.util.Arrays;

/**
 * 第 8 步测试所需的问题实例。
 *
 * <p>本类故意使用与 JSON 完全相同的字段名，使用 Jackson 或 Gson
 * 序列化后无需再做字段名映射。所有贝位编号均从 1 开始。</p>
 */
public class Instance {

    /**
     * 各贝位的作业量。
     * W[0] 对应 1 号贝位，W[i] 对应 i+1 号贝位；每个值必须为非负整数。
     */
    public int[] W;

    /**
     * 桥吊数量。桥吊编号为 1, 2, ..., M，并按物理位置从左到右固定编号。
     */
    public int M;

    /**
     * t=0 时必须开工作业的贝位集合，贝位编号从 1 开始。
     * 可以为空数组，但不能包含重复、相邻或作业量为 0 的贝位。
     */
    public int[] S;

    /**
     * 一次任意距离移动占用的时间槽数，必须为非负整数。
     * 0 表示在相邻作业时段边界瞬时换位；1 是原项目默认语义；
     * k>1 表示移动连续占用 k 个时间槽。
     */
    public int move_time = 1;

    /** JSON 反序列化使用。 */
    public Instance() {
    }

    public Instance(int[] W, int M, int[] S, int move_time) {
        this.W = W;
        this.M = M;
        this.S = S;
        this.move_time = move_time;
    }

    @Override
    public String toString() {
        return "Instance{" +
                "W=" + Arrays.toString(W) +
                ", M=" + M +
                ", S=" + Arrays.toString(S) +
                ", move_time=" + move_time +
                '}';
    }
}

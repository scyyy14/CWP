import java.util.ArrayList;
import java.util.List;

/**
 * 只执行第 8 步时使用的固定源排程。
 *
 * <p>JSON 最少只需包含 move_time、makespan 和 slots。拆分贝位数、负荷、
 * 移动次数等统计值不需要由发送方计算，Python 程序会从 slots 重新计算并
 * 独立校验。时间编号从 0 开始，桥吊和贝位编号从 1 开始。</p>
 *
 * <p>特别说明：move_time=0 时不得创建 state="move" 的记录。每条记录内
 * start_bay 必须等于 end_bay；如果同一桥吊在 time=t 和 time=t+1 的位置
 * 不同，程序就把这次位置变化解释为两个时段边界上的瞬时移动。</p>
 */
public class SourceSchedule {

    /**
     * 源排程采用的移动时间，必须与 Instance.move_time 完全一致。
     * 导师当前“移动时间为 0”的版本应填写 0。
     */
    public int move_time = 1;

    /**
     * 完工时间（时间槽总数）。合法时间编号为 0 到 makespan-1。
     */
    public int makespan;

    /**
     * 完整的逐时间槽桥吊状态。
     * 每个时间槽必须恰好有 M 条记录，因此通常 slots.size() == makespan * M。
     */
    public List<Slot> slots = new ArrayList<>();

    /** JSON 反序列化使用。 */
    public SourceSchedule() {
    }

    public SourceSchedule(int move_time, int makespan, List<Slot> slots) {
        this.move_time = move_time;
        this.makespan = makespan;
        this.slots = slots;
    }

    /**
     * 一台桥吊在一个单位时间槽内的状态。
     */
    public static class Slot {

        /** 时间槽编号，从 0 开始，表示区间 [time, time+1)。 */
        public int time;

        /** 桥吊编号，从 1 开始，取值范围为 1..M。 */
        public int crane;

        /**
         * 状态，只能是 "work"、"move" 或 "idle"。
         * work 表示完成 1 单位作业；move 表示该时间槽正在移动；
         * idle 表示位置不变且不作业。
         */
        public String state;

        /** 时间槽左边界的桥吊位置；作业和停车时为整数贝位。 */
        public double start_bay;

        /** 时间槽右边界的桥吊位置；作业和停车时为整数贝位。 */
        public double end_bay;

        /**
         * state="work" 时为正在作业的 1-based 贝位号；
         * state="move" 或 "idle" 时必须为 null。
         */
        public Integer work_bay;

        /**
         * 仅 move_time>1 的移动槽使用：同一次移动的标识；
         * move_time 为 0 或 1 时填写 null 即可。
         */
        public Integer move_id;

        /**
         * 仅 move_time>1 时使用：当前是该次移动的第几步，从 1 开始；
         * 其他情况填写 null。
         */
        public Integer move_step;

        /**
         * 仅 move_time>1 时使用：该次移动的总步数，应等于 move_time；
         * 其他情况填写 null。
         */
        public Integer move_steps;

        /** JSON 反序列化使用。 */
        public Slot() {
        }

        public Slot(
                int time,
                int crane,
                String state,
                double start_bay,
                double end_bay,
                Integer work_bay
        ) {
            this.time = time;
            this.crane = crane;
            this.state = state;
            this.start_bay = start_bay;
            this.end_bay = end_bay;
            this.work_bay = work_bay;
        }
    }
}

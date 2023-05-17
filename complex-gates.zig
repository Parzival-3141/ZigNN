const std = @import("std");
const dbprint = std.debug.print;

// Future todos:
// Train network that can switch between gates
// i.e.
// input{x,y, 0, 0, 0} = AND(x,y),
// input{x,y, 0, 0, 1} = NAND(x,y),
// input{x,y, 0, 1, 0} = OR(x,y),
// ...

// Write network for a Full-Adder or ALU

pub fn main() PrintError!void {
    var rand = std.rand.DefaultPrng.init(@bitCast(u64, std.time.timestamp()));
    // var rand = std.rand.DefaultPrng.init(420);

    var net = NN.init(rand.random());
    const iterations = 100 * 1000;
    const gate: Gate = .XNOR;
    const data = gate.data();

    try net.dump(data);
    try train(&net, data, iterations);

    dbprint("{s:-^60}\n", .{@tagName(gate)});
    try net.dump(data);
    dbprint("iterations: {d}\n", .{iterations});

    for (data) |t| {
        dbprint("input: {{{d},{d}}} | expected_output: {d} | actual_output: {d}\n", .{ t[0], t[1], t[2], net.evaluate(t[0], t[1]) });
    }
}

pub fn train(net: *NN, data: Gate.TrainingData, iterations: usize) PrintError!void {
    const epsilon = 1e-2;
    const learning_rate = 1e-1;

    for (0..iterations) |_| {
        finite_descent(net, data, epsilon, learning_rate);
    }
}

fn finite_descent(net: *NN, data: Gate.TrainingData, epsilon: f32, learning_rate: f32) void {
    const c = net.cost(data);
    var saved: f32 = undefined;
    var finite_diff: NN = undefined;

    // Find Finite Difference
    inline for (@typeInfo(NN).Struct.fields) |f| {
        saved = @field(net, f.name);
        @field(net, f.name) += epsilon;
        @field(finite_diff, f.name) = (net.cost(data) - c) / epsilon;
        @field(net, f.name) = saved;
    }

    // Update weights
    inline for (@typeInfo(NN).Struct.fields) |f| {
        @field(net, f.name) -= learning_rate * @field(finite_diff, f.name);
    }
}

// Neural Net structure:
// 2 inputs, 3 neurons
//  "OR"  \
//       "AND" -> output
// "NAND" /
//
// "OR", "AND", and "NAND" neurons wont necessarily function like the
// gates they're named after, but that's not the point. We're assuming
// this NN structure is optimal since it mirrors the implementation of
// a XOR gate, which we're trying to emulate.

const NN = struct {
    or_w0: f32,
    or_w1: f32,
    or_b: f32,
    nand_w0: f32,
    nand_w1: f32,
    nand_b: f32,
    and_w0: f32,
    and_w1: f32,
    and_b: f32,

    pub fn init(rand: std.rand.Random) NN {
        return NN{
            .or_w0 = rand.float(f32),
            .or_w1 = rand.float(f32),
            .or_b = rand.float(f32),
            .nand_w0 = rand.float(f32),
            .nand_w1 = rand.float(f32),
            .nand_b = rand.float(f32),
            .and_w0 = rand.float(f32),
            .and_w1 = rand.float(f32),
            .and_b = rand.float(f32),
        };
    }

    pub fn evaluate(self: NN, x: f32, y: f32) f32 {
        const or_value = sigmoid(self.or_w0 * x + self.or_w1 * y + self.or_b);
        const nand_value = sigmoid(self.nand_w0 * x + self.nand_w1 * y + self.nand_b);
        return sigmoid(self.and_w0 * or_value + self.and_w1 * nand_value + self.and_b);
    }

    pub fn cost(self: NN, data: Gate.TrainingData) f32 {
        var y: f32 = undefined;
        var delta: f32 = undefined;

        var err: f32 = 0;
        for (data) |t| {
            y = self.evaluate(t[0], t[1]);
            delta = y - t[2];
            err += delta * delta;
        }

        err /= @as(f32, data.len);
        return err;
    }

    pub fn dump_plot_header() PrintError!void {
        try print("\"or_w0\" \"or_w1\" \"or_b\" \"nand_w0\" \"nand_w1\" \"nand_b\" \"and_w0\" \"and_w1\" \"and_b\" \"cost\"\n", .{});
    }

    pub fn dump_plot(self: NN, data: Gate.TrainingData) PrintError!void {
        try print("{d} {d} {d} {d} {d} {d} {d} {d} {d} {d}\n", .{
            self.or_w0,
            self.or_w1,
            self.or_b,
            self.nand_w0,
            self.nand_w1,
            self.nand_b,
            self.and_w0,
            self.and_w1,
            self.and_b,
            self.cost(data),
        });
    }

    pub fn dump(self: NN, data: Gate.TrainingData) PrintError!void {
        try print("or_w0: {d}\nor_w1: {d}\nor_b: {d}\nnand_w0: {d}\nnand_w1: {d}\nnand_b: {d}\nand_w0: {d}\nand_w1: {d}\nand_b: {d}\ncost: {d}\n", .{
            self.or_w0,
            self.or_w1,
            self.or_b,
            self.nand_w0,
            self.nand_w1,
            self.nand_b,
            self.and_w0,
            self.and_w1,
            self.and_b,
            self.cost(data),
        });
    }
};

// [0..1] = input, [2] = expected
const and_data = Gate.TrainingData{
    .{ 0, 0, 0 },
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 1, 1 },
};

const nand_data = Gate.TrainingData{
    .{ 0, 0, 1 },
    .{ 1, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 1, 0 },
};

const or_data = Gate.TrainingData{
    .{ 0, 0, 0 },
    .{ 1, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 1, 1 },
};

const xor_data = Gate.TrainingData{
    .{ 0, 0, 0 },
    .{ 1, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 1, 0 },
};

const nor_data = Gate.TrainingData{
    .{ 0, 0, 1 },
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 1, 0 },
};

const xnor_data = Gate.TrainingData{
    .{ 0, 0, 1 },
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 1, 1 },
};

const Gate = enum {
    AND,
    NAND,
    OR,
    XOR,
    NOR,
    XNOR,

    const TrainingData = [4][3]f32;

    fn data(self: Gate) TrainingData {
        return switch (self) {
            .AND => and_data,
            .NAND => nand_data,
            .OR => or_data,
            .XOR => xor_data,
            .NOR => nor_data,
            .XNOR => xnor_data,
        };
    }
};

fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

const PrintError = std.fs.File.WriteError;
fn print(comptime fmt: []const u8, args: anytype) PrintError!void {
    var bw = std.io.bufferedWriter(std.io.getStdOut().writer());
    try bw.writer().print(fmt, args);
    try bw.flush();
}

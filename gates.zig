const std = @import("std");
const dbprint = std.debug.print;

// [0..1] = input, [2] = expected
const or_data = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 1, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 1, 1 },
};

const and_data = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 1, 1 },
};

const nand_data = [_][3]f32{
    .{ 0, 0, 1 },
    .{ 1, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 1, 0 },
};

const Gate = enum {
    AND,
    NAND,
    OR,

    fn data(self: Gate) [4][3]f32 {
        return switch (self) {
            .AND => and_data,
            .OR => or_data,
            .NAND => nand_data,
        };
    }
};

fn cost(training_data: [4][3]f32, w0: f32, w1: f32, b: f32) f32 {
    var y: f32 = undefined;
    var delta: f32 = undefined;

    var err: f32 = 0;
    for (training_data) |t| {
        // y = t[0] * w0 + t[1] * w1;
        y = sigmoid(t[0] * w0 + t[1] * w1 + b);
        delta = y - t[2];
        err += delta * delta;
    }

    err /= @as(f32, training_data.len);
    return err;
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

pub fn train(rand: std.rand.Random, gate: Gate) !void {
    var w0 = rand.float(f32);
    var w1 = rand.float(f32);
    var b = rand.float(f32);

    const epsilon = 1e-2;
    const learning_rate = 1e-1;
    const iterations = 100 * 1000;

    const data = gate.data();
    try print("\"w0\" \"w1\" \"b\" \"cost\"\n", .{});
    try print("{d} {d} {d} {d}\n", .{ w0, w1, b, cost(data, w0, w1, b) });
    for (0..iterations) |i| {
        const c = cost(data, w0, w1, b);
        // There's a very obvious matrix pattern here.
        const d0 = (cost(data, w0 + epsilon, w1, b) - c) / epsilon;
        const d1 = (cost(data, w0, w1 + epsilon, b) - c) / epsilon;
        const db = (cost(data, w0, w1, b + epsilon) - c) / epsilon;

        w0 -= learning_rate * d0;
        w1 -= learning_rate * d1;
        b -= learning_rate * db;

        if (i % 500 == 0)
            try print("{d} {d} {d} {d}\n", .{ w0, w1, b, cost(data, w0, w1, b) });
        // try print("w0: {d} w1: {d} b: {d} cost: {d}\n", .{ w0, w1, b, cost(data, w0, w1, b) });
    }

    dbprint("{s:-^60}\n", .{@tagName(gate)});
    dbprint("iterations: {d}\n", .{iterations});
    dbprint("w0: {d} w1: {d} b: {d} cost: {d}\n", .{ w0, w1, b, cost(data, w0, w1, b) });
    for (data) |t| {
        dbprint("input: {d}|{d} expected output: {d} actual output: {d}\n", .{ t[0], t[1], t[2], sigmoid(t[0] * w0 + t[1] * w1 + b) });
    }
}

pub fn main() void {
    var rand = std.rand.DefaultPrng.init(@bitCast(u64, std.time.timestamp()));
    // var rand = std.rand.DefaultPrng.init(420);
    train(rand.random(), .OR) catch {
        dbprint("print failed!\n", .{});
    };
}

const PrintError = std.fs.File.WriteError;
fn print(comptime fmt: []const u8, args: anytype) PrintError!void {
    var bw = std.io.bufferedWriter(std.io.getStdOut().writer());
    try bw.writer().print(fmt, args);
    try bw.flush();
}

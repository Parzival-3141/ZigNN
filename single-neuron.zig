const std = @import("std");
const dbprint = std.debug.print;

// [0] = input, [1] = expected
const training_data = [_][2]f32{
    .{ 0, 0 },
    .{ 1, 2 },
    .{ 2, 4 },
    .{ 3, 6 },
    .{ 4, 8 },
};

fn cost(w: f32) f32 {
    var y: f32 = undefined;
    var d: f32 = undefined;

    var err: f32 = 0;
    for (training_data) |t| {
        y = t[0] * w;
        d = y - t[1];
        err += d * d;
    }
    err /= @as(f32, training_data.len);
    return err;
}

pub fn train_last_braincell(rand: std.rand.Random) !void {
    var w = rand.float(f32) * 10;

    const epsilon = 1e-3;
    const learning_rate = 1e-3;

    try print("\"cost\"\n {d}\n", .{cost(w)});

    for (1000) |_| {
        const delta_cost = (cost(w + epsilon) - cost(w)) / epsilon;
        w -= learning_rate * delta_cost;
        try print("{d}\n", .{cost(w)});
    }

    dbprint("--------------------\n", .{});
    dbprint("weight: {d}\n", .{w});

    for (training_data) |t| {
        dbprint("expected: {d} actual: {d}\n", .{ t[1], t[0] * w });
    }
}

pub fn main() void {
    var rand = std.rand.DefaultPrng.init(@bitCast(u64, std.time.timestamp()));
    // var rand = std.rand.DefaultPrng.init(420);
    train_last_braincell(rand.random()) catch {};
}

fn print(comptime fmt: []const u8, args: anytype) std.fs.File.WriteError!void {
    var bw = std.io.bufferedWriter(std.io.getStdOut().writer());
    try bw.writer().print(fmt, args);
    try bw.flush();
}

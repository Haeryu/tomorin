const std = @import("std");

pub fn build(b: *std.Build) void {
    // const cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\";
    // const cudnn_path = "C:\\Program Files\\NVIDIA\\CUDNN\\v9.7\\";

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tomo = b.dependency("tomo", .{});
    const nina = b.dependency("nina", .{});

    const lib_mod = b.addModule("tomorin", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // lib_mod.addIncludePath(.{ .cwd_relative = "./src/kernel/" });
    // lib_mod.addIncludePath(.{ .cwd_relative = cuda_path ++ "include" });
    // //  lib_mod.addAfterIncludePath(.{ .cwd_relative = cuda_path ++ "include" });
    // lib_mod.addIncludePath(.{ .cwd_relative = cudnn_path ++ "include\\12.8" });
    // // lib_mod.addAfterIncludePath(.{ .cwd_relative = cudnn_path ++ "include\\12.8" });
    // lib_mod.addLibraryPath(.{ .cwd_relative = cuda_path ++ "bin" });
    // lib_mod.addLibraryPath(.{ .cwd_relative = cudnn_path ++ "bin\\12.8" });

    // lib_mod.addLibraryPath(b.path("zig-out/bin"));

    // lib_mod.linkSystemLibrary("cudart64_12", .{});
    // lib_mod.linkSystemLibrary("cublas64_12", .{});
    // lib_mod.linkSystemLibrary("cublasLt64_12", .{});
    // lib_mod.linkSystemLibrary("curand64_10", .{});
    // lib_mod.linkSystemLibrary("cufft64_11", .{});
    // lib_mod.linkSystemLibrary("cudnn64_9", .{});
    // lib_mod.linkSystemLibrary("cusolver64_11", .{});
    // lib_mod.linkSystemLibrary("cusolverMg64_11", .{});
    // lib_mod.linkSystemLibrary("cusparse64_12", .{});
    // lib_mod.linkSystemLibrary("tomo_kernels", .{});
    // lib_mod.addObjectFile(b.path("zig-out/bin/tomo_kernels.lib"));

    lib_mod.addImport("tomo", tomo.module("tomo"));
    lib_mod.addImport("nina", nina.module("nina"));

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.addImport("tomorin", lib_mod);
    exe_mod.addImport("tomo", tomo.module("tomo"));

    const exe = b.addExecutable(.{
        .name = "tomorin",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}

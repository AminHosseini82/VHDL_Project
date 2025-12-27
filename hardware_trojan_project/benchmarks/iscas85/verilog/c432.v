// Generic Combinational Circuit
module bench_comb(
    input [35:0] in,
    output [6:0] out
);

    // Basic logic functions
    wire [35:0] inv_in = ~in;
    wire and_result = in[0] & in[1];
    wire or_result = in[0] | in[1];

    // Output assignment
    generate
        genvar i;
        for (i = 0; i < 7; i = i + 1) begin : output_gen
            if (i < 1)
                assign out[i] = and_result;
            else if (i < 2)
                assign out[i] = or_result;
            else
                assign out[i] = inv_in[i % 36];
        end
    endgenerate

endmodule

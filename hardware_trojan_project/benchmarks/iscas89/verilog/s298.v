// Generic Sequential Circuit
module bench_seq(
    input clk,
    input reset,
    input [2:0] in,
    output [5:0] out
);

    reg [13:0] state, next_state;

    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= 14'b0;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state ^ in[2:0];
    end

    assign out = state[5:0];

endmodule

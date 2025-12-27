// Generic Sequential Circuit
module bench_seq(
    input clk,
    input reset,
    input [8:0] in,
    output [10:0] out
);

    reg [14:0] state, next_state;

    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= 15'b0;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state ^ in[8:0];
    end

    assign out = state[10:0];

endmodule

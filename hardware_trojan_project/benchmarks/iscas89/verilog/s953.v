// Generic Sequential Circuit
module bench_seq(
    input clk,
    input reset,
    input [15:0] in,
    output [0:0] out
);

    reg [28:0] state, next_state;

    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= 29'b0;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state ^ in[15:0];
    end

    assign out = state[0:0];

endmodule

# MyProject/run_tests.jl

# Informacja o rozpoczęciu testów
println("--- Rozpoczynanie testów modułu AutoDiff ---")

# Ścieżka do modułu AutoDiff - zakładamy, że run_tests.jl jest w folderze nadrzędnym do 'autodiff'
# Jeśli uruchamiasz z folderu 'MyProject/', a 'autodiff' jest podfolderem:
include("./autodiff/autodiff.jl")
using .AutoDiff # Używamy zdefiniowanego modułu

# --- Funkcja pomocnicza do ładnego drukowania ---
function print_test_results(test_name::String, variables::Vector{<:Tuple{String, AutoDiff.Node}}, loss_node::AutoDiff.Node)
    println("\n--- Test: $test_name ---")
    println("Wartość straty (loss value): ", AutoDiff.value(loss_node))
    
    # Wyzeruj gradienty przed nowym obliczeniem (jeśli testy są od siebie zależne lub parametry są współdzielone)
    # W tym przypadku każdy test tworzy nowe Node'y, więc nie jest to krytyczne, ale dobra praktyka.
    # params_to_zero = AutoDiff.Node[]
    # for (_, node) in variables
    #     # Jeśli node jest parametrem lub wejściem, dla którego chcemy gradient
    #     # Można by to zrobić bardziej elegancko, zbierając wszystkie Node'y, które są wejściami.
    #     push!(params_to_zero, node)
    # end
    # AutoDiff.zero_grad!(params_to_zero) # To wymaga, aby Node'y miały gradienty zainicjalizowane.
                                         # Dla prostych testów, gradienty są `nothing` przed backward!
                                         # więc zero_grad! może nie być potrzebne lub wymagać dostosowania.

    # Propagacja wsteczna
    try
        AutoDiff.backward!(loss_node)
        println("Gradienty po backward!:")
        for (name, node) in variables
            grad_val = AutoDiff.grad(node)
            if grad_val === nothing
                println("  ∇$(name): Gradient nieobliczony (nothing)")
            else
                println("  ∇$(name): ", grad_val)
            end
        end
    catch e
        println("Błąd podczas backward! dla testu '$test_name': $e")
        # Opcjonalnie: pokaż stacktrace
        # Base.showerror(stdout, e, Base.catch_backtrace())
        # println()
    end
    println("--- Koniec testu: $test_name ---")
end

# --- Definicje Testów ---

# Test 1: Proste dodawanie skalarów
println("\nRozpoczynanie Testu 1: Dodawanie skalarów")
x1 = AutoDiff.Node(2.0f0) # Użyjmy Float32 dla spójności, jeśli planujesz używać w NN
y1 = AutoDiff.Node(3.0f0)
z1 = x1 + y1 # Powinno być 5.0
# Aby mieć od czego liczyć gradient, zdefiniujmy "stratę" jako samą wartość z1
# W rzeczywistości strata byłaby bardziej skomplikowana.
loss1 = z1 
print_test_results("Dodawanie skalarów (z = x+y, loss=z)", [("x1",x1), ("y1",y1)], loss1)
println("Oczekiwane gradienty: ∇x1 = 1.0, ∇y1 = 1.0")

# Test 2: Proste mnożenie skalarów
println("\nRozpoczynanie Testu 2: Mnożenie skalarów")
x2 = AutoDiff.Node(4.0f0)
y2 = AutoDiff.Node(5.0f0)
z2 = x2 * y2 # Powinno być 20.0 (pamiętaj, że * jest teraz aliasem dla .*)
loss2 = z2
print_test_results("Mnożenie skalarów (z = x*y, loss=z)", [("x2",x2), ("y2",y2)], loss2)
println("Oczekiwane gradienty: ∇x2 = 5.0, ∇y2 = 4.0")

# Test 3: Łańcuch operacji skalarnych
println("\nRozpoczynanie Testu 3: Łańcuch operacji")
a3 = AutoDiff.Node(2.0f0)
b3 = AutoDiff.Node(3.0f0)
# c3 = a3 * b3 # c3 = 6
# loss3 = c3 + a3 # loss3 = 6 + 2 = 8
# Rozbijmy to, aby było bardziej zgodne z tym, jak działałby graf
# Użyjmy `.*` jawnie, jeśli tak jest zdefiniowane, lub `*` jeśli jest to alias
op1_3 = a3 .* b3 # Wartość: 2*3=6
loss3 = op1_3 + a3   # Wartość: 6+2=8
print_test_results("Łańcuch operacji (op1 = a*b, loss = op1+a)", [("a3",a3), ("b3",b3)], loss3)
println("Oczekiwane gradienty: ∇a3 = 4.0 (1 od `loss=op1+a` + 3 od `op1=a*b`), ∇b3 = 2.0")

# Test 4: Operacje na tablicach (wektorach) i suma
println("\nRozpoczynanie Testu 4: Operacje na wektorach i suma")
v_a4 = AutoDiff.Node(Float32[1.0, 2.0, 3.0])
v_b4 = AutoDiff.Node(Float32[4.0, 5.0, 6.0])
# v_c4 = v_a4 .* v_b4  # Element-wise: [4.0, 10.0, 18.0]
# loss4 = sum(v_c4)    # Suma: 4 + 10 + 18 = 32.0
# Jawniej:
op1_4 = v_a4 .* v_b4
loss4 = sum(op1_4)
print_test_results("Operacje na wektorach (op1 = Va .* Vb, loss = sum(op1))", [("Va4",v_a4), ("Vb4",v_b4)], loss4)
println("Oczekiwane gradienty: ∇Va4 = [4.0, 5.0, 6.0], ∇Vb4 = [1.0, 2.0, 3.0]")

# Test 5: Mnożenie macierzy i suma
println("\nRozpoczynanie Testu 5: Mnożenie macierzy")
m_a5 = AutoDiff.Node(Float32[1.0 2.0; 3.0 4.0]) # Macierz 2x2
m_b5 = AutoDiff.Node(Float32[5.0 6.0; 7.0 8.0]) # Macierz 2x2
# m_c5 = AutoDiff.matmul(m_a5, m_b5)
# Wartość m_c5: [1*5+2*7  1*6+2*8;  3*5+4*7  3*6+4*8]
#              = [19.0  22.0; 43.0  50.0]
# loss5 = sum(m_c5) # Suma: 19+22+43+50 = 134.0
# Jawniej:
op1_5 = AutoDiff.matmul(m_a5, m_b5)
loss5 = sum(op1_5)
print_test_results("Mnożenie macierzy (op1 = matmul(Ma, Mb), loss = sum(op1))", [("Ma5",m_a5), ("Mb5",m_b5)], loss5)
# Oczekiwane gradienty:
# d(loss)/d(op1_5) = ones(2,2)
# d(loss)/d(m_a5) = d(loss)/d(op1_5) * m_b5' = ones(2,2) * [5 7; 6 8] = [11 13; 11 13]
# d(loss)/d(m_b5) = m_a5' * d(loss)/d(op1_5) = [1 3; 2 4] * ones(2,2) = [4 4; 6 6]
println("Oczekiwane gradienty: ∇Ma5 = [11.0 13.0; 11.0 13.0], ∇Mb5 = [4.0 4.0; 6.0 6.0]") # Transponowane!
# Poprawka: Julia liczy A*B', a my chcemy (dL/dC) * B'. dL/dC to macierz jedynek.
# grad_A = J * B' = [1 1; 1 1] * [5 7; 6 8] = [1*5+1*6  1*7+1*8; 1*5+1*6  1*7+1*8] = [11 15; 11 15]
# grad_B = A' * J = [1 3; 2 4] * [1 1; 1 1] = [1*1+3*1  1*1+3*1; 2*1+4*1  2*1+4*1] = [4 4; 6 6]
println("Poprawione oczekiwane gradienty: ∇Ma5 = [11.0 15.0; 11.0 15.0], ∇Mb5 = [4.0 4.0; 6.0 6.0]")


# Test 6: Funkcje aktywacji ReLU i Sigmoid (na wektorze)
println("\nRozpoczynanie Testu 6: Funkcje aktywacji")
x6 = AutoDiff.Node(Float32[-2.0, 0.0, 3.0])

# Test ReLU
relu_out6 = AutoDiff.relu(x6) # Oczekiwana wartość: [0.0, 0.0, 3.0]
loss_relu6 = sum(relu_out6)    # Oczekiwana strata: 3.0
print_test_results("ReLU (out = relu(x), loss = sum(out))", [("x6_relu",x6)], loss_relu6)
# d(loss)/d(relu_out6) = [1,1,1] (bo sum)
# d(relu_out6)/dx6 = [0,0,1] (pochodna relu)
# d(loss)/dx6 = [1,1,1] .* [0,0,1] = [0,0,1]
println("Oczekiwane gradienty dla ReLU: ∇x6_relu = [0.0, 0.0, 1.0]")

# Potrzebujemy nowego x6 dla testu Sigmoid, bo poprzedni ma już ustawiony gradient
# lub musimy zaimplementować i użyć zero_grad! bardziej rygorystycznie
# Na razie tworzymy nowy Node
x6_sig = AutoDiff.Node(Float32[-2.0, 0.0, 3.0])
sigmoid_out6 = AutoDiff.sigmoid(x6_sig)
# sig(-2) approx 0.1192
# sig(0)  = 0.5
# sig(3)  approx 0.9526
# Oczekiwana wartość: [0.1192, 0.5, 0.9526] (przybliżone)
loss_sigmoid6 = sum(sigmoid_out6) # Oczekiwana strata: ~1.5718
print_test_results("Sigmoid (out = sigmoid(x), loss = sum(out))", [("x6_sigmoid",x6_sig)], loss_sigmoid6)
# Pochodna sig(x) to sig(x)*(1-sig(x))
# s1 = 0.1192, s1*(1-s1) = 0.1192 * 0.8808 = 0.1050
# s2 = 0.5,    s2*(1-s2) = 0.5 * 0.5 = 0.25
# s3 = 0.9526, s3*(1-s3) = 0.9526 * 0.0474 = 0.0452
# Oczekiwane gradienty dla Sigmoid: ∇x6_sigmoid = [0.1050, 0.25, 0.0452] (przybliżone)
println("Oczekiwane gradienty dla Sigmoid: ∇x6_sigmoid = [~0.1050, 0.25, ~0.0452]")


println("\n--- Wszystkie proste testy zakończone ---")
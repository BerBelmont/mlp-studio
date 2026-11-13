"""Interface inicial de menus no terminal."""

OPCOES = {
    "1": "Criar rede",
    "2": "Carregar dataset CSV",
    "3": "Treinar rede",
    "4": "Avaliar rede",
    "5": "Salvar rede",
    "6": "Carregar rede",
    "7": "Realizar previsão",
}


def exibir_menu() -> None:
    """Exibe as operações previstas para a aplicação."""
    print("\n=== Redes Neurais MLP ===")
    for numero, descricao in OPCOES.items():
        print(f"{numero} - {descricao}")
    print("0 - Sair")


def main() -> None:
    """Executa o menu até o usuário sair ou encerrar a entrada do terminal."""
    while True:
        exibir_menu()
        try:
            opcao = input("Escolha uma opção: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAplicação encerrada.")
            return

        if opcao == "0":
            print("Aplicação encerrada.")
            return
        if opcao in OPCOES:
            print(f"{OPCOES[opcao]}: funcionalidade ainda não implementada.")
        else:
            print("Opção inválida. Escolha uma das opções do menu.")

extends Node

var server := WebSocketServer.new()

func _ready():
	server.client_connected.connect(_on_connection)
	server.data_received.connect(_on_data)

	server.listen(8889)
	print("listenting")

func _on_data(id):
	var peer := server.get_peer(id)
	var data = peer.get_packet().get_string_from_utf8()
	match str(data):
		'restart':
			get_tree().get_nodes_in_group("player")[0].position = Vector2(400, 400)
		'state':
			var img := get_viewport().get_texture().get_image().save_png_to_buffer()
			peer.put_packet(img)
		'reward':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			var t: Node2D = get_tree().get_nodes_in_group("target")[0]
			var r: float = p.position.distance_to(t.position)
			if r > 512 or p.position.x < 0 or p.position.y < 0 or p.position.y > 512 or p.position.x > 512:
				r = -10.0
			elif r < 50:
				r = 10.0
			else:
				r = -1 * pow((r / 512 * 10), 3)
			var a = PackedByteArray()
			a.resize(4)
			a.encode_float(0, r)
			print(r)
			peer.put_packet(a)
		'0':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.y -= 10
		'1':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.x += 10
		'2':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.y += 10
		'3':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.x -= 10
		'4':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.y -= 10
			p.position.x -= 10
		'5':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.x += 10
			p.position.y -= 10
		'6':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.y += 10
			p.position.x -= 10
		'7':
			var p: Node2D = get_tree().get_nodes_in_group("player")[0]
			p.position.x += 10
			p.position.y += 10






func _on_connection(id, proto, name):
	var peer := server.get_peer(id)
	var img := get_viewport().get_texture().get_image().save_png_to_buffer()
	print(get_viewport().get_texture().get_size())

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _physics_process(delta):
	server.poll()
